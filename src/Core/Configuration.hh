/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
// $Id$

#ifndef _CORE_CONFIGURATION_HH
#define _CORE_CONFIGURATION_HH

#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include <iostream>
#include <list>
#include <set>
#include <vector>
#include <string.h>
#include <sys/stat.h>
#include "Assertions.hh"
#include "ReferenceCounting.hh"
#include "Types.hh"
#include "XmlStream.hh"

namespace Core {

class AbstractParameter;

/**
 * @page Configuration
 *
 * All resources from
 *  - environment variables
 *  - configuration files
 *  - the command line
 *
 * are kept within an @c Configuration object. Each configurable
 * module (derived form @c Configurable) may ask this class for
 * resources using a parameter specification string of the form
 *
 * <selector1>.<selector2>. ... .<selectorN>.<name>
 *
 * A resources consists of a name and an associated value.  The name
 * of the resource has the form
 *
 * <selector1>.<selector2>. ... .<selectorN>
 *
 * where any of the selectors may by the wildcard "*".  The values
 * are stored as strings.  Conversion to appropriate type (flag,
 * bool, int, float, choice, ...) is done by parameter declaration
 * classes derive from Parameter.
 *
 * @see Core::Configuration
 * @see Core::Configurable
 * @see Core::Parameter
 *
 * @subsection Resource References
 * The value of a resource may contain a reference of the form
 * $(selector).  When this resource is looked up (i.e. matched
 * against a parameter specification), the reference is textually
 * replaced by its resolved value.  The resolved value is determined
 * by appending the resources selector to the matched parameter
 * specification and searching for a matching resource.  This implies
 * that resources are @em context @em dependent. If this failed the
 * matched parameter specification is iteratively truncated until
 * either a match is found or the resolution fails.  Example:
 * Given the resources
 * *.foo.abc   = cat
 * *.foo.xyz   = dog
 * foo.*.bar = /tmp/$(foo).txt
 * then looking up
 * foo.abc.bar     -> /tmp/cat.txt
 * foo.xyz.bar     -> /tmp/dog.txt
 * @see Core::Configuration::resolve()
 */

/**
 * Central configuration class.
 * @ref Configuration
 */
class Configuration {
public:
    static const char  resource_wildcard_char;
    static const char* resource_wildcard_string;
    static const char  resource_separation_char;
    static const char* resource_separation_string;

    /**
     * Test if argument is a well-formed resource name.
     * A string is a well-formed resource name iff it matches the
     * regular grammar: S   ->   '*' |  [a-zA-Z0-1]+  |  S '.' S
     */
    static bool isWellFormedResourceName(const std::string&);

    /**
     * Test if argument is a well-formed parameter name.
     * A string is a well-formed resource name iff it matches the
     * regular grammar: S   ->   [a-zA-Z0-1]+  |  S '.' S
     */
    static bool isWellFormedParameterName(const std::string&);

    static std::string prepareResourceName(const std::string& selection, const std::string& name);

    class SourceDescriptor;
    class Resource;
    class ResourceDataBase;

protected:
    Ref<ResourceDataBase> db_;

private:
    std::string selection_;
    std::string name_;

    void warning(const std::string& filename, int lineNumber,
                 const std::string& description);
    void error(const std::string& filename, int lineNumber,
               const std::string& description);

    /**
     *  Parses include.
     *  Syntax: include filename
     *  Include can be used within a selection as well. In this case,
     *  the including selection will be active until the first selection in the included file.
     */
    bool parseInclude(const std::string& line,
                      std::string&       includeFilename,
                      std::string&       warningDescription);
    bool parseSelection(const std::string& line,
                        std::string&       selection,
                        std::string&       warningDescription);
    bool parseComment(const std::string& line,
                      std::string&       warningDescription);
    bool parseResource(const std::string& line,
                       std::string& parameter, std::string& value,
                       std::string& warningDescription);

    static bool              isWellFormedName(const std::string&, bool allowWildcards);
    bool                     setFromFile(const std::string& filename,
                                         const std::string& currentSelection);
    std::vector<std::string> setFromCommandline(
            const std::vector<std::string>& arguments,
            const SourceDescriptor*         source);

public:
    // pseudo copy
    Configuration(const Configuration& c, const std::string& selection);
    Configuration(const Configuration& c);
    Configuration();
    ~Configuration();

    Configuration& operator=(const Configuration& c);

    /**
     * Enable logging facility.
     **/
    void enableLogging();

    /**
     * Report where configuration ressources came from.
     **/
    void writeSources(XmlWriter&) const;

    /**
     * Report all configuration resolved resources in the database.
     */
    void writeResolvedResources(XmlWriter&) const;

    /**
     * Report all configuration resources in the database.
     */
    void writeResources(XmlWriter&) const;

    /**
     * Return all configuration resources in command-line format
     */
    std::vector<std::string> writeResourcesToCommandlineArgs() const;

    /**
     * Report how configuration ressources where used by which
     * parameters.
     **/
    void writeUsage(XmlWriter&) const;

    const std::string& getSelection() const {
        return selection_;
    };

    const std::string& getName() const {
        return name_;
    };

    // manipulate configuration context
    void setSelection(const std::string& selection);

    /**
     * Add a resource.
     * If a resource with the same name already exists, its value is
     * replaced.
     * @param name of the resource to be added
     * @param value of the resource
     */
    void set(const std::string&      name,
             const std::string&      value  = "true",
             const SourceDescriptor* source = 0);

    /**
     * Try to add a resource.
     * This is a safe veriosn of set(). If the resource name is not
     * well-formed, an error message is printed and no resource is set.
     * @return true is @c name is well-formed.
     * @see set() */
    bool tryToSet(const std::string&      name,
                  const std::string&      value  = "true",
                  const SourceDescriptor* source = 0);

    bool                     setFromEnvironment(const std::string& variable);
    bool                     setFromFile(const std::string& filename);
    std::vector<std::string> setFromCommandline(const std::vector<std::string>& arguments);

    /**
     * Query the configuration for a parameter.
     *
     * It is strongly discouraged to use get() directly use one of the
     * Parameter classes instead. @see Parameter
     *
     * @param parameter is the parameter identification string
     * @param value The value of the queried parameter is stored in @c value.
     * Remains unchanged if the parameter is not specified in the
     * configuration.
     * @return true if the parameter was configured
     **/
    bool get(const std::string& parameter, std::string& value) const;

    /**
     * Substitutes all parameter references, arithmetic expressions
     * and cache manager commands in a string. Cache manager commands
     * are only resolved if the corresponding module is enabled
     *
     * @see Configuration::resolveReferences
     * @see Configuration::resolveArithmeticExpressions
     * @see Configuration::resolveCacheManagerCommands
     */
    std::string resolve(const std::string& value) const;

    /**
     * Substitute arithmetic expressions in a string
     *
     * All occurances of $[expression] are substituted.
     * expression can be any valid arithmetic expression, including
     * standard math funcions.
     * To cast the result of the expression to a specific type, use
     * $[expression,format]
     * Format may be int or float, default is float.
     * Parameter references are not resolved! So use resolveReferences
     * before.
     *
     * @param value is a string which may contain arithmetic expressions
     * @return @vc value with all arith. expressions subsituted
     */
    static std::string resolveArithmeticExpressions(const std::string& value);

private:
    /**
     * Find the resource for a given parameter.
     * @param parameter the parameter specification string
     * @return the most specific resource matching @c parameter
     */
    const Resource* find(const std::string& parameter) const;
    /**
     * Substitute parameter references in a string.
     *
     * All occurances of $(config) are substituted based on the
     * configuration.  resolve() works recursively if necessary.
     * resolve() will first try to find a resource for the reference
     * expanded by the current selection (local lookup).  If that
     * fails the selection if succesively shortened until the
     * unexpanded string is tried (global lookup).  (So we try
     * A.B.C.REF, A.B.REF, A.REF and REF in this order until one of
     * them is found.)  @warning Note that the result of local lookup
     * depends on the Configuration which initiates the resolution!
     *
     * resolve() is intended for transformation of strings given by the
     * user.  Do NOT use resolve() like this: resolve("$(fudge)"); use
     * one of the Parameter classes instead. @see Parameter
     *
     * @param value is a string which may contain configuration references.
     * @return @c value with all references substituted.
     **/
    std::string resolveReferences(const std::string& value) const;

#ifdef MODULE_CORE_CACHE_MANAGER
    /**
     * Substitute cache manager commands in a string
     *
     * A cache manager command is enclosed by "`cf" and "`". Arguments are
     * separated by whitespace (no escape characters). If the "-d" flag is
     * given the local file will be copied to the origin when the application
     * terminates.
     * This should be called after Configuration::resolveReferences and
     * Configuration::resolveArithmeticExpressions
     *
     * @param value is a string which may contain cache manager commands.
     * @return @c value with all cache manager commands substituted.
     **/
    std::string resolveCacheManagerCommands(const std::string& value) const;
#endif

    std::string getResolvedValue(const Resource* resource) const;
};

/**
 * Describes where a resource comes from.
 */
class Configuration::SourceDescriptor {
public:
    std::string type, data;
    void        write(XmlWriter& os) const {
        os << XmlFull("source", data) + XmlAttribute("type", type);
    }
};

/**
 * Item of configuration.
 *
 * A resource is a piece of configuration specified by the user.
 * It consists of a name and an associated value.  The name may
 * contain wildcards.  The value may contain references
 * (e.g. $(basedir) ), which are subject to substitution.
 */
class Configuration::Resource {
private:
    const SourceDescriptor* source_;
    std::string             name_;
    std::string             value_;
    mutable bool            isBeingResolved_; /**< flag to trap circular reference */

    struct Usage {
        std::string              fullParameterName;
        const AbstractParameter* parameter;
        std::string              effectiveValue;
    };

    mutable std::vector<Usage> usage;

public:
    Resource(const std::string& _name, const std::string& _value, const SourceDescriptor* _source)
            : source_(_source),
              name_(_name),
              value_(_value),
              isBeingResolved_(false) {}

    inline const std::string& getName() const {
        return name_;
    };
    inline const std::string& getValue() const {
        return value_;
    };

    bool isBeingResolved() const {
        return isBeingResolved_;
    }
    void beginResolution() const {
        isBeingResolved_ = true;
    }
    void endResolution() const {
        isBeingResolved_ = false;
    }

    inline bool operator<(const Resource& r) const {
        return name_ < r.name_;
    }
    inline bool operator==(const Resource& r) const {
        return name_ == r.name_;
    }

    void write(std::ostream& os) const {
        os << name_ << " = " << value_;
    }

    /**
     * Determine if the resource matches a configurtion path.
     * @param components the components of the configuration path
     * @return the number of path components matched by the resource,
     * or -1 of the resource does not match.
     */
    s32 match(const std::vector<std::string>& components) const;

    void registerUsage(const std::string& n, const AbstractParameter* p, const std::string& v) const {
        Usage u;
        u.fullParameterName = n;
        u.parameter         = p;
        u.effectiveValue    = v;
        usage.push_back(u);
    }

    void writeUsage(XmlWriter&) const;
};

/**
 * Central storage place for all resources.
 */
class Configuration::ResourceDataBase : public ReferenceCounted {
private:
    std::set<Resource> resources;
    Resource           noResource_;
    bool               isLogging_;

    typedef std::list<SourceDescriptor*> SourceList;
    SourceList                           sources_;

public:
    /**
     * Add a resource.
     * If a resource with the same name already exists, its value is
     * replaced.
     * @param name of the resource to be added
     * @param value of the resource
     */
    void set(const std::string&      name,
             const std::string&      value  = "true",
             const SourceDescriptor* source = 0);

    /**
     * Find the resource to be used for a given parameter.
     * @param parameter the parameter specification string
     * @return the most specific resource matching @c parameter
     */
    const Resource* find(const std::string& parameter) const;

    const std::set<Resource>& getResources() const {
        return resources;
    }

    const Resource* noResource() const {
        return &noResource_;
    }

    SourceDescriptor* addSource(const std::string& type, const std::string& data) {
        SourceDescriptor* sd = new SourceDescriptor;
        sd->type             = type;
        sd->data             = data;
        sources_.push_back(sd);
        return sd;
    }

    ResourceDataBase();
    ~ResourceDataBase();

    void enableLogging() {
        isLogging_ = true;
    }
    void writeSources(XmlWriter&) const;
    void writeUsage(XmlWriter&) const;
    void write(std::ostream&) const;
};

}  // namespace Core

#endif  // _CORE_CONFIGURATION_HH
