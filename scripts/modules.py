#! /usr/bin/env python
""" Tool for handling of Sprint modules """

__version__ = "$Id$"
__author__  = "David Rybach"

import sys, os
import optparse
DEBUG = False

def debug(msg):
    if DEBUG:
        sys.stderr.write("DEBUG: %s\n" % str(msg))

class ModInfo:
    def __init__(self):
        self.makefiles = []
        self.files = []

class FileCutter:
    def __init__(self, enabledModules, release = False):
        self.enabledModules = enabledModules
        self.release = release

    def stripFile(self, inputFilename, outputFilename):
        self.initialize()
        self.outputFile = open(outputFilename, "wt")
        for line in open(inputFilename, "rt"):
            sLine = line.strip()
            if self.shallParseLine(sLine):
                if self.handleDirective(sLine):
                    self.output(line)
                else:
                    self.output(line, True)
            else:
                if self.isActive():
                    self.output(line)
                else:
                    self.output(line, True)

    def initialize(self):
        self.conStack = [ (True,"", False) ]

    def isActive(self):
        active = True
        for item in self.conStack:
            active = active and item[0]
        debug("isActive %s -> %d" % (str([ i[0] for i in self.conStack]), active))
        return active

    def output(self, line, remove = False):
        if not remove:
            print(line.rstrip())

    def pushCon(self, con, directive, isModule):
        debug("pushCon(%s, %s, %d)" % (con, directive, isModule))
        self.conStack.append( (con, directive, isModule) )

    def shallParseLine(self, line):
        raise Exception("not implemented")


class MakeFileCutter (FileCutter):

    def output(self, line, remove = False):
        if not remove:
            self.outputFile.write(line.rstrip() + "\n")
        elif not self.release:
            print('# removed: ' + line.rstrip())

    def shallParseLine(self, line):
        r = (len(line) and line.split()[0] in [ 'ifdef', 'ifndef', 'endif', 'ifeq', 'ifneq', 'else', 'elif' ])
        if r: debug(line)
        return r

    def handleDirective(self, sLine):
        tokens = sLine.split()
        value = None
        if tokens[0] == 'ifdef':
            val, ism = self.evalIfdef(tokens[1])
            self.pushCon(val, "", ism)
            retval = self.isActive()
        elif tokens[0] == 'ifndef':
            val, ism = self.evalIfdef(tokens[1])
            self.pushCon(not val, "", ism)
            retval = self.isActive()
        elif tokens[0] == 'ifeq':
            self.pushCon(self.isActive(), "", False)
            retval = self.isActive()
        elif tokens[0] == 'ifneq':
            self.pushCon(self.isActive(), "", False)
            retval = self.isActive()
        elif tokens[0] == 'else':
            prev = self.conStack.pop()
            if prev[2]:
                self.pushCon(not prev[0], "", prev[2])
                if self.isActive():
                    self.output("ifeq (1,1)")
                return False
            else:
                self.pushCon(True, "", False)
                retval = self.isActive()
        elif tokens[0] == 'endif':
            retval = self.isActive()
            self.conStack.pop()
        else:
            retval = self.isActive()
        debug("handleDirective(" + sLine + ") -> " + str(retval))
        return retval

    def evalIfdef(self, token):
        if token[0:7] != "MODULE_":
            return True, False
        else:
            return (token in self.enabledModules), True


class SourceFileCutter (FileCutter):

    def output(self, line, remove = False):
        if not remove:
            self.outputFile.write(line.rstrip() + "\n")
        elif not self.release:
            print('/* removed: ' + line.rstrip() + ' */')

    def shallParseLine(self, line):
        return (len(line) and line[0] == '#')

    def handleDirective(self, sLine):
        tokens = sLine.split()
        negate = False
        value = None
        if tokens[0] == '#ifdef' or tokens[0] == '#ifndef':
            val, sym, ism = self.evalIfdef(tokens[0], tokens[1])
            self.pushCon(val, sym, ism)
            retval = self.isActive()
        elif tokens[0] == '#if':
            val, ism = self.evalIf(tokens[1:])
            self.pushCon(val, " ".join(tokens[1:]), ism)
            retval = self.isActive()
        elif tokens[0] == '#else':
            debug("handleElse")
            prev = self.conStack.pop()
            debug("prev" + str(prev))
            if not prev[2]:
                self.pushCon(True, "! (%s)" % prev[1], False)
                retval = self.isActive()
            else:
                wasActive = self.isActive()
                self.pushCon(not prev[0], "! (%s)" % prev[1], True)
                retval = False
                if wasActive and prev[0]:
                    self.output("#endif")
                elif wasActive:
                    if self.release:
                        self.output("#if 1")
                    else:
                        self.output("#if 1 /* ! (%s) */" % prev[1])
        elif tokens[0] == "#elif":
            prev = self.conStack.pop()
            if not prev[2]:
                self.pushCon(True, "", False)
                return True
            elif not prev[0]:
                active, ism = self.evalIf(tokens[1:])
                self.pushCon(active, " ".join(tokens[1:]), ism)
                if self.isActive():
                    self.output("#if %s" % " ".join(tokens[1:]))
            else:
                if self.isActive():
                    self.output("#endif")
                self.pushCon(False, " ".join(tokens[1:]), True)
            retval = False
        elif tokens[0] == '#endif':
            retval = self.isActive()
            self.conStack.pop()
        else:
            retval = self.isActive()
        debug("handleDirective: %s -> %s %s" % (sLine, retval, self.conStack))
        return retval

    def evalIfdef(self, ifdef, token):
        debug("evalIfdef(%s, %s)" % (ifdef, token))
        if ifdef == "#ifndef":
            neg = "!"
        else:
            neg = ""
        sym = "%s defined(%s)" % (neg, token)
        if token[0:7] != "MODULE_":
            return True, sym, False
        else:
            val = (token in self.enabledModules)
            if ifdef == "#ifndef":
                val = not val
            return val, sym, True

    def evalIf(self, tokens):
        s = []
        tokens = " ".join(tokens).replace("!", " ! ").replace("(", " ( ").replace(")", " ) ").split()
        i = 0
        isModule = True
        defSeen = False
        comment = False
        debug("evalIf: " + str(tokens))
        while i < len(tokens):
            t = tokens[i]
            if comment:
                if t == "*/":
                    comment = False
            elif t == '&&':
                s += [ 'and' ]
            elif t == '||':
                s += [ 'or' ]
            elif t == 'defined':
                defSeen = True
                tmp = self.evalIfdef('ifdef', tokens[i+2])
                isModule = isModule and tmp[2]
                s += [ str(tmp[0]) ]
                i += 3
            elif t == "!":
                s += [ "not" ]
            elif t == "(" or t == ")":
                s += [ t ]
            elif t == '/*':
                comment = True
            elif t == '0':
                s += [ "False" ]
            else:
                s += [ "True" ]
            i += 1
        debug("s: " + str(s))
        isModule = (isModule and defSeen) or (len(tokens) == 1 and (tokens[0] == '0' or tokens[0] == '1'))
        if isModule:
            ret = eval(" ".join(s))
        else:
            ret = True
        return ret, isModule


def parseMakefile(filename, modFiles):
    curMod = []
    basedir = os.path.dirname(os.path.realpath(filename))
    for line in open(filename, "rt"):
        line = line.strip()
        if line == "":
            continue
        aline = line.split()
        if aline[0] == "ifdef":
            curMod.append(aline[1])
            # print("curMod: ", str(curMod))
        elif aline[0] == "endif" and len(curMod) > 0:
            curMod.pop()
            # print("curMod: ", str(curMod))
        else:
            if len(curMod) > 0:
                curFile = ""
                if len(aline) > 1 and aline[1] == '+=' and aline[0].split("_")[-1] == "O":
                    curFile = "/".join(aline[2].split("/")[1:])
                    # print("added: ", str(modFiles))
                elif aline[0] == "#MODF":
                    curFile = aline[1]
                if curFile != "":
                    for mod in curMod:
                        if not modFiles.has_key(mod):
                            modFiles[mod] = ModInfo()
                        f = basedir + "/" + os.path.basename(curFile)
                        modFiles[mod].files.append(f)
                        if not filename in modFiles[mod].makefiles:
                            modFiles[mod].makefiles.append(filename)
    return modFiles

def parseModulesMake(filename):
    disabled = []
    enabled = []
    for line in open(filename, "rt"):
        line = line.strip()
        if line == "":
            continue
        aline = line.split()
        if len(aline) < 3:
            continue
        if aline[0] == '#' and aline[1] == "MODULES":
            disabled.append(aline[3])
        elif aline[0] == "MODULES" and aline[1] == "+=":
            enabled.append(aline[2])
    sys.stderr.write("enabled  modules:\n%s\n" % str(enabled))
    sys.stderr.write("disabled modules:\n%s\n" % str(disabled))
    return disabled, enabled

def addSourceFiles(modFiles):
    for mod in modFiles.keys():
        newarr = []
        for item in modFiles[mod].files:
            name, e = os.path.splitext(item)
            if e == ".o":
                for ext in ["cc","hh"]:
                    f = name + "." + ext
                    if os.path.exists(f):
                        newarr.append(f)
            else:
                newarr.append(item)
        modFiles[mod].files = newarr

    print("addSourceFiles: ", [  (k, len(v.files)) for k, v in modFiles.items() ])


def main(options, args):
    modFiles = {}
    for makefile in args:
        # sys.stderr.write("parse %s\n" % makefile)
        parseMakefile(makefile, modFiles)
    addSourceFiles(modFiles)
    disabledModules, enabledModules = [], []

    if options.modulesMake != None:
        if (not options.action in [ "check", "strip" ]) and \
           (not (options.enabledModules or options.disabledModules)):
            print("Error: you have to specify whether you want to use")
            print("       disabled or enabled modules from Modules.make")
            return False

        disabledModules, enabledModules = parseModulesMake(options.modulesMake)
        if options.enabledModules:
            options.modules = enabledModules
        else:
            options.modules = disabledModules

    if options.action == "files":
        print("options.modules: ", options.modules)
        for module in options.modules:
            try:
                for f in modFiles[module].files:
                    print(f)
            except KeyError:
                pass
    elif options.action == "modules":
        modules = modFiles.keys()
        modules.sort()
        for m in modules:
            print(m)
    elif options.action == "check":
        if options.modulesMake == None:
            print("Error: no Modules.make specified")
            return False
        foundModules = modFiles.keys()
        foundModules.sort()
        definedModules = enabledModules + disabledModules
        for m in modFiles.keys():
            if not m in definedModules:
                print("Warning: unspecified module %s in %s" % (m, str(modFiles[m].makefiles)))
    elif options.action == "rename":
        print(options)
        for module in options.modules:
            try:
                for f in modFiles[module].files:
                    nf = os.path.dirname(f) + "/" + options.prefix + os.path.basename(f)
                    print(f, " -> ", nf)
                    try:
                        os.rename(f, nf)
                    except Exception as e:
                        sys.stderr.write("error: %s\n" % str(e))
            except KeyError:
                pass
    elif options.action == "strip":
        if options.modulesMake == None:
            print("Error: no Modules.make specified")
            return False
        sf = SourceFileCutter(enabledModules, options.release)
        mf = MakeFileCutter(enabledModules, options.release)
        for filename in options.processFiles.split(","):
            if not len(filename): continue
            oldfile = filename + ".bak"
            sys.stderr.write(filename + " -> " + oldfile + "\n")
            os.rename(filename, oldfile)
            aFilename = filename.split(".")
            if aFilename[-1] in [ 'cc', 'hh', 'c', 'h', 'tcc' ]:
                sys.stderr.write('source file: %s\n' % filename)
                sf.stripFile(oldfile, filename)
            else:
                sys.stderr.write('make   file: %s\n' % filename)
                mf.stripFile(oldfile, filename)
    else:
        print("Error: unknown action: ", options.action)
        return False
    return True


if __name__ == "__main__":
    actions = "actions:\n"\
              "  modules      print all modules found in the given makefiles\n"\
              "  files        print filenames of given modules\n"\
              "  rename       rename files of given modules to _<filename>\n"\
              "  check        check files for consitency\n"\
              "  strip        remove code for disabled modules from source files and makefiles\n"

    optparser = optparse.OptionParser(usage = "%prog [OPTIONS] <Makefiles>\n" + actions)
    optparser.add_option("-a", "--action", help="action [default: modules]", default="modules")
    optparser.add_option("-m", "--module", help="set module(s)", action="append", dest="modules", default=[])
    optparser.add_option("-p", "--prefix", help="prefix used for renaming [default: _]", default="_")
    optparser.add_option("-f", "--modules-file", help="Modules.make", default=None,
                         dest="modulesMake")
    optparser.add_option("-e", "--enabled", help="get enabled modules from Modules.make",
                         default=False, action="store_true", dest="enabledModules")
    optparser.add_option("-d", "--disabled", help="get disabled modules from Modules.make",
                         default=False, action="store_true", dest="disabledModules")
    optparser.add_option("-s", "--process-file", help="files to process (comma separated list) for action 'strip'",
                         default="", dest="processFiles")
    optparser.add_option("-r", "--release", help="do not put debug information into striped code",
                         default=False, action="store_true", dest="release")

    options, args = optparser.parse_args()
    if len(args) < 1:
        optparser.print_help()
        sys.exit(1)
    main(options, args)




