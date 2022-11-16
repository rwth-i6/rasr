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
#include <cstdlib>
#include <string>
#include <ctype.h>
#include <pwd.h>
#include <sys/resource.h>
#include <sys/utsname.h>
#include <unistd.h>
#ifdef OS_linux
#include <malloc.h>
#endif
#include <Modules.hh>
#include "Application.hh"
#include "CacheManager.hh"
#include "Channel.hh"
#include "Configuration.hh"
#include "Directory.hh"
#include "MappedArchive.hh"
#include "MemoryInfo.hh"
#include "Parameter.hh"
#include "Statistics.hh"
#include "Version.hh"

extern char** environ;

using namespace Core;

const ParameterBool   Application::paramLogConfiguration("log-configuration", "log-configuration", false);
const ParameterBool   Application::paramLogResolvedResources("log-resolved-resources", "log-resolved-resources", false);
const ParameterString Application::paramConfig("config", "configuration file");
const ParameterString Application::paramCacheArchiveFile("file", "cache archive file");
const ParameterBool   Application::paramCacheArchiveReadOnly("read-only", "whether the cache archive is read-only", false);
const ParameterBool   Application::paramHelp("help", "help", false);

std::string  Application::path_     = ".";
std::string  Application::basename_ = "";
Application* Application::app_      = 0;

static Configuration& getConfig() {
    static Configuration config;
    return config;
}

Application::Application()
        : Component(getConfig()),
          channelManager_(0),
          debugChannel_(0),
          debugXmlChannel_(0),
          defaultLoadConfigurationFile_(true),
          defaultOutputXmlHeader_(true),
          comment_("#;%") {
    require(!app_);  // There must be only one instance.
    app_ = this;

    char*          value;
    const char*    user;
    struct passwd* p = NULL;

    // load process environment into configuration
    for (const char* const* env = environ; *env; ++env) {
        const char* sep = strchr(*env, '=');
        if (sep) {
            std::string key   = std::string(*env, sep - *env);
            std::string value = std::string(sep + 1);
            config.set(key, value);
        }
    }

    // get the real (!) home directory
    if ((user = getlogin()) != NULL) {
        p = getpwnam(user);
    }
    if (p != NULL) {
        value = p->pw_dir;
    }
    else {
        value = getenv("HOME");
    }

    if (value) {
        config.set("*.home", value);
    }
    else {
        config.set("*.home", "");
    }
}

Application::~Application() {
    app_ = NULL;
    delete debugChannel_;
    debugChannel_ = 0;
    delete debugXmlChannel_;
    debugXmlChannel_ = 0;
    for (std::map<std::string, MappedArchive*>::const_iterator it = cacheArchives_.begin();
         it != cacheArchives_.end(); ++it)
        delete it->second;
    cacheArchives_.clear();
}

Core::Channel& Application::debugChannel() {
    // create debug channel
    if (debugChannel_ == 0) {
        debugChannel_ = new Core::Channel(config, "debug", Core::Channel::standard);
    }
    if (!(*debugChannel_).isOpen())
        error("binary debug channel could not be opened.");
    return *debugChannel_;
}

Core::XmlChannel& Application::debugXmlChannel() {
    // create debug channel
    if (debugXmlChannel_ == 0) {
        debugXmlChannel_ = new Core::XmlChannel(config, "debug", Core::Channel::standard);
    }
    if (!(*debugXmlChannel_).isOpen())
        error("XML debug channel could not be opened.");
    return *debugXmlChannel_;
}

void Application::setDefaultLoadConfigurationFile(bool f) {
    defaultLoadConfigurationFile_ = f;
}

void Application::setDefaultOutputXmlHeader(bool f) {
    defaultOutputXmlHeader_ = f;
}

void Application::setTitle(const std::string& title) {
    config.setSelection(title);
    getConfig().setSelection(title);
    ensure(name() == title);
}

std::string Application::getVariable() const {
    std::string env(name());
    for (size_t i = 0; i < env.length(); i++)
        env[i] = toupper(env[i]);
    return env;
}

std::string Application::getBaseName() const {
    return Application::basename_;
}

std::string Application::getPath() const {
    return Application::path_;
}

std::string Application::getApplicationDescription() const {
    return "usage: " + getBaseName() + " [OPTIONS(S)]" + "\n";
}

std::string Application::getParameterDescription() const {
    return std::string();
}

std::string Application::getDefaultParameterDescription() const {
    std::ostringstream sout;
    paramHelp.printShortHelp(sout);
    paramConfig.printShortHelp(sout);
    sout << std::endl;
    sout << "default channels:" << std::endl;
    sout << "  the default channels are set by" << std::endl;
    sout << "      --xxx.channel <dest>" << std::endl;
    sout << "  where 'xxx' serves as a placeholder for the channel's name." << std::endl;
    sout << "  The destination <dest> of a channel can be 'stdout', 'stderr',"
         << "  'nil' or an arbitrary file name;" << std::endl;
    sout << "  'nil' suppresses any output." << std::endl;
    sout << "  At least the channels 'log', 'warn' and 'err' are supported." << std::endl;
    return sout.str();
}

std::string Application::getUsage() const {
    std::ostringstream sout;

    sout << std::endl
         << getApplicationDescription() << std::endl;

    std::string param = getParameterDescription();
    if (!param.empty())
        sout << "options" << std::endl
             << param << std::endl;
    std::string defaultParam = getDefaultParameterDescription();
    if (!defaultParam.empty())
        sout << "default options" << std::endl
             << defaultParam << std::endl;

    return sout.str();
}

void Application::setCommentCharacters(const std::string& c) {
    if (c.length() == 0)
        return;
    comment_ = c;
}

void Application::logSystemInfo() const {
    XmlChannel c(config, "system-info");
    if (!c.isOpen())
        return;

    struct utsname info;
    if (uname(&info) < 0) {
        warning("failed to determine system information");
    }
    else {
        c << XmlOpen("system-information")
          << XmlFull("name", info.nodename)
          << XmlFull("type", info.machine)
          << XmlFull("operating-system", info.sysname)
          << XmlFull("build-date", __DATE__)
          << XmlFull("local-time", getTime(Component::LogTimingYes))
          << XmlClose("system-information");
    }
}

void Application::logVersion() const {
    VersionRegistry vr;
    XmlChannel      vc(config, "version");
    if (vc.isOpen()) {
        vr.reportVersion(vc);
    }
}

void Application::logResources() const {
    XmlChannel rc(config, "configuration");
    if (!rc.isOpen())
        return;
    rc << XmlOpen("configuration");
    config.writeSources(rc);
    config.writeResources(rc);
    if (paramLogResolvedResources(config))
        config.writeResolvedResources(rc);
    rc << XmlClose("configuration");
}

void Application::logResourceUsage() const {
    XmlChannel rc(config, "configuration-usage");
    if (!rc.isOpen())
        return;
    rc << XmlOpen("configuration")
       << XmlOpen("resources");
    config.writeUsage(rc);
    rc << XmlClose("resources")
       << XmlClose("configuration");
}

void Application::logMemoryUsage() const {
    XmlChannel rc(config, "memory-usage");
    if (!rc.isOpen())
        return;
    MemoryInfo info;
    rc << XmlOpen("virtual-memory")
       << XmlFull("current", info.size())
       << XmlFull("peak", info.peak())
       << XmlClose("virtual-memory");
}

void Application::reportLowLevelError(const std::string& msg) {
    lowLevelErrorMessages_.push_back(msg);
}

void Application::exit(int status) {
    runAtExitFuncs();
    if (channelManager_)
        channelManager_->flushAll();
    std::cerr << "exiting..." << std::endl;
    ::exit(status);
}

void Application::atexit(std::function<void()> f) {
    atExitFuncs_.push_back(f);
}

void Application::runAtExitFuncs() {
    std::list<std::function<void()>> list;
    list.swap(atExitFuncs_);
    for (std::function<void()> f : list)
        f();
};

int Application::run(const std::vector<std::string>& arguments) {
    openLogging();
    int status = main(arguments);
    runAtExitFuncs();
    closeLogging();
    return status;
}

void Application::openLogging() {
    if (paramLogConfiguration(config))
        config.enableLogging();
    channelManager_ = new Channel::Manager(select("channels"), defaultOutputXmlHeader_);

    logSystemInfo();
    logVersion();
    logResources();

    timer_.start();
}

void Application::closeLogging(bool configAvailable) {
    if (lowLevelErrorMessages_.size()) {
        Message                  m(error("There were %d low level error messages:",
                        static_cast<int>(lowLevelErrorMessages_.size())));
        std::vector<std::string> messages = std::vector<std::string>(lowLevelErrorMessages_);
        for (std::vector<std::string>::const_iterator message = messages.begin(); message != messages.end(); ++message)
            m << "\n"
              << message->c_str();
    }
    logMemoryUsage();
    timer_.stop();
    if (configAvailable) {
        logResourceUsage();
        XmlChannel tc(config, "time");
        timer_.write(tc);
    }
    delete channelManager_;
    channelManager_ = 0;
}

bool Application::setMaxStackSize(size_t new_size_in_mb) {
    struct rlimit rl;
    int           success;
    const rlim_t  new_size_in_bytes = new_size_in_mb * 1024 * 1024;
    success                         = getrlimit(RLIMIT_STACK, &rl);
    if (success == 0) {
        if (rl.rlim_cur < new_size_in_bytes) {
            rl.rlim_cur = new_size_in_bytes;
            success     = setrlimit(RLIMIT_STACK, &rl);
            return success == 0;
        }
        else {
            return true;  // no need to increase
        }
    }
    return false;  // can't read = can't write
}

int Application::main(int argc, char* argv[]) {
    /**
     * @bug This is a workaround for possible bug in LIBC 2.3.2. We are waiting for an answer on our bug report.
     */
#ifdef OS_linux
    mallopt(M_TOP_PAD, 1024 * 1024);  // 128*1024 default for GLIBC2.3.2 on debian sarge
#endif

    if (!app_) {
        std::cerr << "oops! you did not instantiate any application class!" << std::endl;
        return EXIT_FAILURE;
    }

    int status = EXIT_SUCCESS;

    std::vector<std::string> arguments;
    for (int i = 0; i < argc; ++i) {
        arguments.push_back(argv[i]);
    }

    setFromCommandline(arguments);

    std::string file = paramConfig(getConfig());
    if (file.empty() && app_->defaultLoadConfigurationFile_)
        file = app_->name() + ".config";
    if (!file.empty())
        setFromFile(file);

    setFromEnvironment(app_->getVariable());

    arguments = setFromCommandline(arguments);

    if (app_->name() == "UNNAMED") {
        std::string title = app_->getBaseName();
        title.erase(title.find("."));
        app_->setTitle(title);
    }
    if (!setMaxStackSize(64)) {
        app_->warning("failed to increase max stack size");
    }

    if (paramHelp(getConfig())) {
        std::cerr << app_->getUsage();
    }
    else {
        status = app_->run(arguments);
#ifdef MODULE_CORE_CACHE_MANAGER
        copyLocalCacheFiles();
#endif
    }

    // Do an early reset of this variable, so that code in exit handlers can check for it.
    // (E.g. Core::printWarning().)
    app_ = NULL;

    return status;
}

MappedArchiveReader Application::getCacheArchiveReader(const std::string& archive, const std::string& item) {
    log() << "requesting reader for item " << item << " in archive " << archive;
    MappedArchive* arch = getCacheArchive(archive);
    if (arch)
        return arch->getReader(item);
    else
        return MappedArchiveReader();
}

MappedArchiveWriter Application::getCacheArchiveWriter(const std::string& archive, const std::string& item) {
    log() << "requesting writer for item " << item << " in archive " << archive;
    MappedArchive* arch = getCacheArchive(archive);
    if (arch)
        return arch->getWriter(item);
    else
        return MappedArchiveWriter();
}

MappedArchive* Application::getCacheArchive(const std::string& archive) {
    if (archive.empty())
        return 0;

    if (cacheArchives_.count(archive))
        return cacheArchives_[archive];

    Core::Configuration keyConfig(config, archive);

    std::string    file     = paramCacheArchiveFile(keyConfig);
    bool           readOnly = paramCacheArchiveReadOnly(keyConfig);
    MappedArchive* ret      = 0;

    log() << "opening cache archive " << archive << " file \"" << file << "\" read-only " << readOnly;

    if (!file.empty())
        ret = new MappedArchive(file, readOnly);

    cacheArchives_[archive] = ret;
    return ret;
}

void Application::updateCacheArchive(const std::string& archive, const Core::Configuration& extraConfig, bool reset) {
    if (archive.empty())
        return;

    if (reset) {
        if (cacheArchives_.count(archive)) {
            delete cacheArchives_[archive];
            cacheArchives_.erase(archive);
        }
        return;
    }

    Core::Configuration keyConfig(extraConfig, archive);
    std::string         file     = paramCacheArchiveFile(keyConfig);
    bool                readOnly = paramCacheArchiveReadOnly(keyConfig);

    log() << "update cache archive " << archive << " file \"" << file << "\" read-only " << readOnly;

    // Note: we always create new ones (reload) in case cache content changed for the same file
    if (cacheArchives_.count(archive))
        delete cacheArchives_[archive];

    MappedArchive* ret = 0;
    if (!file.empty())
        ret = new MappedArchive(file, readOnly);
    cacheArchives_[archive] = ret;
}

bool Application::setFromEnvironment(const std::string& variable) {
    return getConfig().setFromEnvironment(variable);
}

bool Application::setFromFile(const std::string& filename) {
    return getConfig().setFromFile(filename);
}

std::vector<std::string> Application::setFromCommandline(const std::vector<std::string>& arguments) {
    basename_ = baseName(arguments[0]);
    path_     = directoryName(arguments[0]);
    return getConfig().setFromCommandline(arguments);
}
