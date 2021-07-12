#! /usr/bin/env python3
""" Tool for handling of Sprint modules """

__version__ = "$Id$"
__author__  = "rybach@i6.informatik.rwth-aachen.de"

import sys, os, copy
import optparse
import pickle
import re
from modules import parseModulesMake
DEBUG = False

IGNORE_DIRS = ["Tools", "Translation", ".svn", ".build", "doc",
               "development", "Flac", "Test" ]
IGNORE_FILES = [ "check.cc", "check-xml.cc" ]
IGNORE_DEP = [ ("Audio", "Wav") ]
DIR_MODULES = { "Nn": "MODULE_NN",
                "Flf": "MODULE_FLF",
                "Flf/FlfCore": "MODULE_FLF_CORE",
                "Math/Nr": "MODULE_MATH_NR",
                "Search/Wfst": "MODULE_SEARCH_WFST",
                "Search/AdvancedTreeSearch": "MODULE_ADVANCED_TREE_SEARCH",
                "OpenFst": "MODULE_OPENFST",
                "Test": "MODULE_TEST"
              }
""" modules without actual files, used to enable/disable
certain functionality """
PSEUDO_MODULES = [ "MODULE_TBB",
                   "MODULE_OPENMP",
                   "MODULE_INTEL_MKL",
                   "MODULE_ACML",
                   "MODULE_CUDA"
                 ]

""" mapping from special include files to required modules """
DEFAULT_DEPS = { '<omp.h>' : "MODULE_OPENMP",
                 '<acml.h>' : "MODULE_ACML",
                 '"mkl.h"' : "MODULE_INTEL_MKL",
                 '<cublas_v2.h>' : "MODULE_CUDA",
                 '<cuda_runtime.h>' : "MODULE_CUDA",
                 '<tbb/parallel_for.h>' : "MODULE_TBB"
               }

INCLUDE_EXT = [ ".hh", ".cc", ".h", ".tcc", ".c" ]

def debug(msg):
    if DEBUG:
        sys.stderr.write("DEBUG: %s\n" % str(msg))


def findFiles():
    """find all relevant source files"""
    fileList = []
    makefiles = []
    # ignore files with 2 or more dots in the filename
    ignoreRe = re.compile("([^/]+\.[^/]*\.[^/]+)$")
    for root, dirs, files in os.walk("src"):
        debug("root=%s dirs=%s" % (str(root), str(dirs)))
        for dir in IGNORE_DIRS:
            if dir in dirs:
                dirs.remove(dir)
                debug("ignore %s" % dir)
        if root != "src" and root[-3:] != "doc":
            for f in files:
                ext = f[f.rfind("."):]
                pathName = root[4:]
                if ext in INCLUDE_EXT and not f in IGNORE_FILES and \
                        not ignoreRe.match(f):
                    id = len(fileList)
                    fileList.append( (pathName, f) )
                elif (f == "Makefile"):
                    debug("found makefile: %s %s" % (pathName, f))
                    makefiles.append( (pathName, f) )
    return fileList, makefiles


def getEntities(files):
    """create entity ids for a list of filenames"""
    entitities = []
    entityDict = {}
    for p, f in files:
        name = f[:f.rfind(".")]
        entity = (p, name)
        if not entity in entityDict:
            entityDict[entity] = len(entitities)
            entitities.append(entity)
    for m in PSEUDO_MODULES:
        entity = ("#", m)
        if not entity in entityDict:
            entityDict[entity] = len(entitities)
            entitities.append(entity)
    return entitities, entityDict


class Modules:
    """handles module to file relationships.
       parses all makefiles to get the module
       membership information
    """
    def __init__(self, entityDict):
        self.entityDict = entityDict
        self.moduleFiles = []
        self.moduleNames = []
        self.moduleId = {}
        self.entityToModules = []
        self.autoNames = set()

    def parseMakefiles(self, makefiles):
        """read all makefiles and create the set of entities
           for each discovered module
        """
        for p, n in makefiles:
            filename = os.path.join("src", p, n)
        self._parseFile(p, filename)

    def addDefaultModules(self, entities):
        """add entities, which are not part of a pre-defined
           module to a default module MODULE_<DIRNAME>
        """
        self._setupEntityToModules()
        for e in range(len(self.entityDict)):
            path, name = entities[e]
            if self.entityToModules[e] is None:
                isDefaultName = False
                if path in DIR_MODULES:
                    moduleName = DIR_MODULES[path]
                elif path == "#":
                    moduleName = name
                else:
                    moduleName = "MODULE_" + path.upper().replace("/", "_")
                    isDefaultName = True
                m = self._addToModule(moduleName, e)
                self.entityToModules[e] = m
                if isDefaultName: self.autoNames.add(m)
                debug("default: %d -> %s, %d" % (e, self.moduleNames[m], isDefaultName))
        for m in DIR_MODULES.values():
            self._addModule(m)

    def _setupEntityToModules(self):
        for e in range(len(self.entityDict)):
                self.entityToModules.append(None)
        ignore = [ self.entityDict[e] for e in IGNORE_DEP ]
        for m in range(len(self.moduleFiles)):
            for e in self.moduleFiles[m]:
                if e in ignore: continue
                if not self.entityToModules[e]:
                    self.entityToModules[e] = m
                    debug("e2m: %d -> %s" % (e, self.moduleNames[m]))
                else:
                    if self.entityToModules[e] != m:
                        raise Exception("%s != %s" % (self.moduleNames[m],
                            self.moduleNames[self.entityToModules[e]]))

    def _parseFile(self, dir, filename):
        debug("parsing %s" % filename)
        l = 0
        curMod = []
        for line in open(filename, "rt"):
            l += 1
            line = line.strip()
            if not line: continue
            aline = line.split()
            if aline[0] == "ifdef":
                if curMod:
                    sys.stderr.write("ambigous definition in %s:%d\n" % (filename, l))
                    sys.stderr.write("curMod=%s\n" % " ".join(curMod))
                    sys.stderr.write(line + "\n")
                curMod.append(aline[1])
            elif aline[0] == "endif":
                if curMod: curMod.pop()
            else:
                if curMod:
                    try:
                        e = self._getEntity(aline, dir)
                    except KeyError as e:
                        sys.stderr.write("Warning: unknown file in line '%s' (%s:%d)\n" %
                                (line, filename, l))
                        e = -1
                    if e >= 0:
                        assignedModule = self._addToModule(curMod[0], e)
                        debug("%d -> %s" % (e, self.moduleNames[assignedModule]))
                    else:
                        debug("unknown entity in %s: %s" % (filename, str(aline)))

    def _getEntity(self, line, dir):
        curFile = ""
        curDir = dir
        if len(line) > 1 and line[1] == '+=':
            target = line[0].split("_")
            if target[-1] == "O" and target[0] != "CHECK" and \
                    not "libSprint" in line[2]:
                    curFile = "/".join(line[2].split("/")[1:])
        elif line[0] == "#MODF":
            curFile = line[1]
        result = -1
        if curFile:
            entity = (curDir, curFile.split(".")[0])
            debug(str(entity))
            result = self.entityDict[entity]
        return result

    def _addModule(self, module):
        if not module in self.moduleId:
            id = len(self.moduleId)
            self.moduleId[module] = id
            self.moduleNames.append(module)
            self.moduleFiles.append([])
        else:
            id = self.moduleId[module]
        return id

    def _addToModule(self, module, entity):
        id = self._addModule(module)
        self.moduleFiles[id].append(entity)
        return id

    def toStrings(self, modules):
        """list of module ids to list of module names"""
        r = [ self.moduleNames[m] for m in modules ]
        r.sort()
        return r

    def toString(self, module):
        """module id to module name"""
        return self.moduleNames[module]



class Dependencies:
    """handles file dependencies.
       parses all source files to check for included
       header files.
    """
    def __init__(self, modules, entities):
        self.modules = modules
        self.entities = entities
        self.dependencies = {}
        self.connections = {}
        self.fileDependencies = {}
        for m in range(len(modules.moduleNames)):
            self.dependencies[m] = [set(), set()]

    def parseFiles(self, entityDict):
        """parse all source files for all entities in entityDict"""
        for name, id in entityDict.items():
            filename = "src/%s/%s" % name
            includes = []
            for ext in INCLUDE_EXT:
                if os.path.isfile(filename + ext):
                    includes += self._parseFile(filename + ext)
            entities = self._getEntities(entityDict, name, includes)
            self._addDependencies(id, entities)

    def getFileDependencies(self, e):
        """returns direct and indirect module dependencies of the given entity."""
        deps = set()
        for m in self.fileDependencies[e]:
            deps.add(m)
            deps.update(self.dependencies[m][0])
        return self.fileDependencies[e], deps.difference(self.modules.autoNames)

    def _addDependencies(self, e, includes):
        module = self.modules.entityToModules[e]
        debug("module: %d %s" % (module, self.modules.moduleNames[module]))
        for i, cond in includes:
            c = int(cond)
            depModule = self.modules.entityToModules[i]
            debug("depends: %d %s cond=%d" % (depModule, self.modules.moduleNames[depModule], c))
            self.dependencies[module][c].add(depModule)
            if not e in self.fileDependencies:
                self.fileDependencies[e] = set()
            self.fileDependencies[e].add(depModule)
            self._addConnection(module, depModule, e, None)

    def _addConnection(self, modFrom, modTo, entity, module):
        if not modFrom in self.connections:
            self.connections[modFrom] = {modFrom: []}
        if not modTo in self.connections[modFrom]:
            self.connections[modFrom][modTo] = []
        self.connections[modFrom][modTo].append((entity, module))

    def getConnection(self, modFrom, modTo):
        """returns the entities that cause the dependency
           between the two given modules."""
        try:
            item = self.connections[modFrom][modTo]
        except KeyError:
            return None
        ret = []
        for i in item:
            if i[0] is None:
                ret.append(self.modules.moduleNames[i[1]])
            else:
                ret.append(self.entities[i[0]])
        return ret

    def _stripExt(self, f):
        return re.sub("\.hh?$", "", f)

    def _getEntities(self, entityDict, file, includes):
        eIncludes = []
        for name, con in includes:
            n = None
            debug("name: " + name)
            if name in DEFAULT_DEPS:
                n = ("#", DEFAULT_DEPS[name])
            elif name[0] == '"':
                a = (file[0] + "/" + name[1:-1]).split("/")
                n = ("/".join(a[:-1]), self._stripExt(a[-1]))
            elif name.find("/"):
                a = name[1:-1].split("/")
                n = ("/".join(a[:-1]), self._stripExt(a[-1]))
            else:
                debug("ignored include: " + name)
            if n:
                debug("n: " + str(n))
                # remove .hh/.h
                n[1] .replace(".hh", "").replace(".h", "")
                if n in entityDict:
                    debug("%s %d con=%d" % (str(n), entityDict[n], con))
                    eIncludes.append((entityDict[n], con))
                else:
                    debug("unknown include: " + str(n))
        return eIncludes

    def _parseFile(self, filename):
        includes = []
        condStack = [ False ]
        debug("parsing " + filename)
        for line in open(filename, "rt", encoding="utf-8"):
            line = line.strip()
            if not line or line[0] != '#': continue
            sline = line.split()
            if sline[0][:3] == "#if":
                pos = line.find("MODULE_")
                if pos >= 0 and line[pos-1:pos+9] != "_MODULE_HH":
                    condStack.append(True)
                else:
                    condStack.append(condStack[-1])
            elif sline[0] == "#endif":
                condStack.pop()
            elif sline[0] == "#include":
                includes.append((sline[1], condStack[-1]))
        return includes

    def buildClosure(self):
        """build the dependency closure for all known modules.
           resolves the indirect dependencies of all modules.
           updates self.dependencies.
        """
        self.finished = [ False ] * len(self.dependencies)
        for m in self.dependencies.keys():
            trace = [m]
            deps = self._close(trace, m)
            assert(self.dependencies[m][0] == deps)
        del self.finished

    def _close(self, trace, module):
        if self.finished[module]:
            return self.dependencies[module][0]
        deps = copy.copy(self.dependencies[module][0])
        for m in self.dependencies[module][0]:
            if not m in trace:
                c = self._close(trace + [m], m)
                deps.update(c)
                for conn in c:
                    self._addConnection(module, conn, None, m)
        self.dependencies[module][0] = deps
        self.finished[module] = True
        return deps

    def removeDefaultModules(self):
        """removes the automatically generated modules from the
           dependency sets.
        """
        for m in self.dependencies:
            self.dependencies[m][0] = self.dependencies[m][0].difference(self.modules.autoNames)
            if m in self.dependencies[m][0]: self.dependencies[m][0].remove(m)

    def factorized(self, module, cond):
        """returns the factorized set of module dependencies for
           the given module.
           removes module dependencies if they are implied by another
           dependent module.
        """
        newdeps = set()
        todo = copy.copy(self.dependencies[module][cond])
        while todo:
            best = -1
            bestm = -1
            for m in todo:
                if m == module: continue
                intersection = len(todo.intersection(self.dependencies[m][cond]))
                if intersection > best:
                    best = intersection
                    bestm = m
            assert(bestm != -1)
            newdeps.add(bestm)
            todo.difference_update(self.dependencies[bestm][cond])
            if best == 0:
                newdeps.update(todo)
                break
        return newdeps


def checkDependencies(db, enabled, disabled):
    """check if all dependencies are met in the given
       set of enabled modules.
    """
    enabledModules = set()
    for m in enabled:
        try:
                enabledModules.add(db.getModuleId(m))
        except KeyError:
            print("Warning: Unknown module \"%s\"" % m)
    retval = 0
    for mid in enabledModules:
        debug("module %s %d" % (db.getModuleName(mid), mid))
        depModules = db.deps.dependencies[mid][0]
        if not depModules.issubset(enabledModules):
            print("unmet dependencies for module ", db.getModuleName(mid))
            for d in depModules:
                if not d in enabledModules:
                    print("  %s requires %s" % (db.getModuleName(mid), db.getModuleName(d)))
                    retval = 1
    return retval


class Database:
    """creates, loads, stores the Modules and the
       Dependencies object.
    """
    def __init__(self):
        self.modules = None
        self.entities = None
        self.entityDict = None
        self.deps = None

    def write(self, filename):
        fp = file(filename, "wb")
        pickle.dump(self, fp, 2)

    def load(self, filename):
        try:
            fp = file(filename, "rb")
            tmp = pickle.load(fp)
        except Exception as e:
            sys.stderr.write("db load error: %s\n", str(e))
            return False
        self.modules = tmp.modules
        self.entities = tmp.entities
        self.entityDict = tmp.entityDict
        self.deps = tmp.deps
        return True

    def create(self, files, makefiles):
        self.entities, self.entityDict = getEntities(files)
        self.modules = Modules(self.entityDict)
        self.modules.parseMakefiles(makefiles)
        self.modules.addDefaultModules(self.entities)
        self.deps = Dependencies(self.modules, self.entities)
        self.deps.parseFiles(self.entityDict)
        self.deps.buildClosure()
        self.deps.removeDefaultModules()

    def getFileId(self, filename):
        a = filename.replace("src/", "").split("/")
        n = ("/".join(a[:-1]), a[-1][:-3])
        return self.entityDict[n]

    def getEntity(self, fileid):
        return self.entities[fileid]

    def getEntities(self, fileids):
        return [ self.entities[id] for id in fileids ]

    def getModuleId(self, module):
        return self.modules.moduleId[module]

    def getModuleName(self, moduleId):
        return self.modules.moduleNames[moduleId]


def main(options, args):
    db = Database()
    if options.load:
        sys.stderr.write("loading database %s\n" % options.database)
        ok = db.load(options.database)
        if not ok:
            return 1
    else:
        os.chdir(options.basedir)
        files, makefiles = findFiles()
        sys.stderr.write("%d files, %d makefiles\n" % (len(files), len(makefiles)))
        db.create(files, makefiles)
        if options.write:
            sys.stderr.write("storing database %s\n" % options.database)
            db.write(options.database)

    if options.show:
        mids = db.deps.dependencies.keys()
        mids.sort(key = lambda i: db.modules.toString(i))
        for m in mids:
            d = db.deps.dependencies[m]
            if not (d[0] or d[1]): continue
            print(db.modules.toString(m))
            if options.factorize:
                moduleDep = db.deps.factorized(m, 0)
                condModuleDep = db.deps.factorized(m, 1)
            else:
                moduleDep = d[0]
                condModuleDep = d[1]
            print("depends on:", db.modules.toStrings(moduleDep))
            print("cond. depends on:", db.modules.toStrings(condModuleDep))
            print("")
    if options.depfile:
        id = db.getFileId(options.depfile)
        direct, indirect = db.deps.getFileDependencies(id)
        print(id, db.getEntity(id))
        print("direct: ", db.modules.toStrings(direct))
        print("indirect: ", db.modules.toStrings(indirect))
    if options.connection:
        m1, m2 = options.connection.split(",")
        m1 = db.getModuleId(m1)
        m2 = db.getModuleId(m2)
        print(db.getModuleName(m1), "->", db.getModuleName(m2))
        print(db.deps.getConnection(m1, m2))
    if options.moduledep:
        mid = db.getModuleId(options.moduledep)
        print(mid, db.getModuleName(mid))
        if options.factorize:
            moduleDep = db.deps.factorized(mid, 0)
            condModuleDep = db.deps.factorized(mid, 1)
        else:
            moduleDep, condModuleDep = db.deps.dependencies[mid]
        print("depends on: ", db.modules.toStrings(moduleDep))
        print("cond. depends on: ", db.modules.toStrings(condModuleDep))
    if options.files:
        mid = db.getModuleId(options.files)
        print(mid, db.getModuleName(mid))
        print(db.getEntities( db.modules.moduleFiles[mid] ))
    if options.check:
        disabled, enabled = parseModulesMake(options.modulesMake)
        return checkDependencies(db, enabled, disabled)
    return 0


if __name__ == "__main__":
    optparser = optparse.OptionParser(usage = "%prog [OPTIONS]")
    optparser.add_option("-k", "--modules-file", help="Modules.make", default="Modules.make",
                         dest="modulesMake")
    optparser.add_option("-a", "--database", help="database file", default="deps.db",
                            dest="database")
    optparser.add_option("-l", "--load", help="load database", default=False, action="store_true",
                         dest="load")
    optparser.add_option("-w", "--write", help="store database", default=False, action="store_true",
                         dest="write")
    optparser.add_option("-s", "--show", help="show all dependencies", default=False, action="store_true",
                         dest="show")
    optparser.add_option("-d", "--dependencies", help="dependencies of a file", default=None,
                         dest="depfile")
    optparser.add_option("-c", "--connection", help="dependencies between 2 modules", default=None,
                         dest="connection")
    optparser.add_option("-m", "--module", help="module dependencies", default=None,
                         dest="moduledep")
    optparser.add_option("-t", "--factorize", help="factorize dependencies", default=False, action="store_true",
                         dest="factorize")
    optparser.add_option("-f", "--files", help="show files of a module", default=None,
                         dest="files")
    optparser.add_option("-e", "--check", help="check if module dependencies are fulfilled "
                                               "with the modules enabled in Modules.make",
                         action="store_true", default=False, dest="check")
    optparser.add_option("-b", "--basedir", help="package root directory",
                         default=".", dest="basedir")

    options, args = optparser.parse_args()
    r = main(options, args)
    sys.exit(r)

