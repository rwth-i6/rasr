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
#include "Debug.hh"

#include <stdio.h>

#ifdef OS_linux
#include <execinfo.h>
#endif

#if defined(__APPLE__)
#include <cassert>
#include <stdbool.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <unistd.h>
#elif defined(OS_linux)
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace Core {

void printWarning(const char* msg, ...) {
    va_list ap;
    va_start(ap, msg);
    auto* app = Core::Application::us();
    if (app)
        Core::Application::us()->vWarning(msg, ap);
    else {
        fprintf(stderr, "App instance not available. Global warning: ");
        vfprintf(stderr, msg, ap);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    va_end(ap);
}

void printLog(const char* msg, ...) {
    va_list ap;
    va_start(ap, msg);
    auto* app = Core::Application::us();
    if (app)
        Core::Application::us()->vLog(msg, ap);
    else {
        fprintf(stderr, "App instance not available. Global log: ");
        vfprintf(stderr, msg, ap);
        fprintf(stderr, "\n");
        fflush(stderr);
    }
    va_end(ap);
}

#if defined(__APPLE__)

// Based on Apple's recommended method as described in
// http://developer.apple.com/qa/qa2004/qa1361.html
bool AmIBeingDebugged()
// Returns true if the current process is being debugged (either
// running under the debugger or has a debugger attached post facto).
{
    // Initialize mib, which tells sysctl what info we want.  In this case,
    // we're looking for information about a specific process ID.
    int mib[] =
            {
                    CTL_KERN,
                    KERN_PROC,
                    KERN_PROC_PID,
                    getpid()};

    // Caution: struct kinfo_proc is marked __APPLE_API_UNSTABLE.  The source and
    // binary interfaces may change.
    struct kinfo_proc info;
    size_t            info_size = sizeof(info);

    int sysctl_result = sysctl(mib, sizeof(mib) / sizeof(*mib), &info, &info_size, NULL, 0);
    if (sysctl_result != 0)
        return false;

    // This process is being debugged if the P_TRACED flag is set.
    return (info.kp_proc.p_flag & P_TRACED) != 0;
}

#elif defined(OS_linux)

// just assume Linux-like with /proc

bool AmIBeingDebugged() {
    // We can look in /proc/self/status for TracerPid.  We are likely used in crash
    // handling, so we are careful not to use the heap or have side effects.
    int status_fd = open("/proc/self/status", O_RDONLY);
    if (status_fd == -1)
        return false;

    // We assume our line will be in the first 1024 characters and that we can
    // read this much all at once.  In practice this will generally be true.
    // This simplifies and speeds up things considerably.
    char buf[1024];

    ssize_t num_read     = read(status_fd, buf, sizeof(buf));
    buf[sizeof(buf) - 1] = 0;  // safety
    close(status_fd);
    if (num_read <= 0)
        return false;

    const char* searchStr = "TracerPid:\t";
    const char* f         = strstr(buf, searchStr);
    if (f == NULL)
        return false;

    // Our pid is 0 without a debugger, assume this for any pid starting with 0.
    f += strlen(searchStr);
    return f < &buf[num_read] && *f != '0';
}

#else

// TODO implement
bool AmIBeingDebugged() {
    return false;
}
#warning No AmIBeingDebugged implementation for your platform.

#endif

void stackTrace(std::ostream& os, int cutoff) {
#ifdef OS_linux
#if defined(DEBUG) || defined(SPRINT_DEBUG_LIGHT)
    static const size_t maxTraces = 100;

    // Get backtrace lines
    void*  array[maxTraces];
    size_t nTraces = backtrace(array, maxTraces);
    char** strings = backtrace_symbols(array, nTraces);

    // Extract addresses
    const char* tmpNamBuf1 = "./bt-addresses.tmp";
    const char* tmpNamBuf2 = "./bt-results.tmp";

    std::ofstream out(tmpNamBuf1);
    for (size_t i = cutoff + 1; i < nTraces; ++i) {
        std::string line(strings[i]);
        size_t      firstAddrPos = line.find("[") + 1;
        size_t      addrLength   = line.find("]" - firstAddrPos);
        out << line.substr(firstAddrPos, addrLength) << std::endl;
    }
    out.close();

    // Run addr2line to get usable stack trace information
    Core::Application* thisApp      = Core::Application::us();
    std::string        addr2lineCmd = "addr2line -C -f -e " + thisApp->getPath() + "/" + thisApp->getBaseName() + " < " + tmpNamBuf1 + " > " + tmpNamBuf2;

    os << std::endl
       << "Analyzing stack trace with command " << addr2lineCmd << std::endl;
    os << "Please be patient (approx. 30 s)..." << std::endl;
    system(addr2lineCmd.c_str());

    // Evaluate addr2line output
    os << "Stack trace (innermost first):" << std::endl;
    std::ifstream in(tmpNamBuf2);
    for (size_t i = cutoff + 1; i < nTraces; ++i) {
        std::string line1;
        getline(in, line1);
        std::string line2;
        getline(in, line2);
        std::string line(strings[i]);
        size_t      firstAddrPos = line.find("[");
        size_t      addrLength   = line.find("]") - firstAddrPos + 2;
        os << "#" << i << ":\t" << line1 << std::endl;
        os << "\t   at: " << line2 << " " << line.substr(firstAddrPos, addrLength) << std::endl;
    }

    // Clean up.
    free(strings);
    unlink(tmpNamBuf1);
    unlink(tmpNamBuf2);

#else
    os << "Creating stack trace (innermost first):" << std::endl;
    static const size_t maxTraces = 256;
    void*               array[maxTraces];
    size_t              nTraces = backtrace(array, maxTraces);
    char**              strings = backtrace_symbols(array, nTraces);
    for (size_t i = cutoff + 1; i < nTraces; i++)
        os << '#' << i << "  " << strings[i] << std::endl;
    free(strings);
#endif  // DEBUG
#endif  // OS_LINUX
}

}  // namespace Core
