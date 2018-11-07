/** Copyright 2018 RWTH Aachen University. All rights reserved.
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

#include "Assertions.hh"

#include <errno.h>
#ifdef OS_linux
#include <execinfo.h>
#endif
#include <cstdio>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "Application.hh"
#include "Debug.hh"

using namespace Core;

namespace AssertionsPrivate {

static int PUTS(int fd, const char* s) {
    return write(fd, s, (int)strlen(s));
}

void safe_stackTrace(int fd) {
#ifdef OS_linux
    PUTS(fd, "Creating stack trace (innermost first):\n");
    static const size_t maxTraces = 100;
    void*               array[maxTraces];
    size_t              nTraces = backtrace(array, maxTraces);
    backtrace_symbols_fd(array, nTraces, fd);
#endif
}

void abort() __attribute__((noreturn));

FailedAssertion::FailedAssertion(const char*  type,
                                 const char*  expr,
                                 const char*  function,
                                 const char*  filename,
                                 unsigned int line) {
    std::cerr << std::endl
              << std::endl
              << "PROGRAM DEFECTIVE:"
              << std::endl
              << type << ' ' << expr << " violated" << std::endl
              << "in " << function
              << " file " << filename << " line " << line << std::endl;
}

FailedAssertion::~FailedAssertion() {
    std::cerr << std::endl;
    stackTrace(std::cerr, 1);
    std::cerr << std::endl
              << std::flush;
    abort();
}

void assertionFailed(const char*  type,
                     const char*  expr,
                     const char*  function,
                     const char*  filename,
                     unsigned int line) {
    FailedAssertion(type, expr, function, filename, line);
}

void hopeDisappointed(const char*  expr,
                      const char*  function,
                      const char*  filename,
                      unsigned int line) {
    std::cerr << std::endl
              << std::endl
              << "RUNTIME ERROR:"
              << std::endl
              << "hope " << expr << " disappointed" << std::endl
              << "in " << function
              << " file " << filename << " line " << line;
    if (errno)
        std::cerr << ": " << strerror(errno);
    std::cerr << std::endl
              << std::endl;
    stackTrace(std::cerr, 1);
    std::cerr << std::endl
              << "PLEASE CONSIDER ADDING PROPER ERROR HANDLING !!!" << std::endl
              << std::endl
              << std::flush;
    abort();
}

class ErrorSignalHandler {
    static volatile sig_atomic_t isHandlerActive;
    static void                  handler(int);

public:
    ErrorSignalHandler();
    void abort() __attribute__((noreturn));
};

volatile sig_atomic_t ErrorSignalHandler::isHandlerActive = 0;

void ErrorSignalHandler::handler(int sig) {
    if (!isHandlerActive) {
        isHandlerActive = 1;
        std::cerr << std::endl
                  << std::endl
                  << "PROGRAM DEFECTIVE (TERMINATED BY SIGNAL):" << std::endl
                  << strsignal(sig) << std::endl
                  << std::endl;
        stackTrace(std::cerr, 1);
        std::cerr << std::endl
                  << std::flush;
    }
    signal(sig, SIG_DFL);
    raise(sig);
}

static void notifyHandler(int sig) {
    // This handler is called for notification (e.g. on SIGUSR1).
    // We want to continue with normal operation when the signal handler finished,
    // so we need to be more careful here (in contrast to handler()).
    // See signal() manpage about what is safe to call.
    // Basically, avoid any malloc() and most other dynamic things.
    PUTS(STDERR_FILENO, "\n\nRECEIVED NOTIFICATION SIGNAL:\n");
    PUTS(STDERR_FILENO, strsignal(sig));
    PUTS(STDERR_FILENO, "\n\n");
    safe_stackTrace(STDERR_FILENO);
    PUTS(STDERR_FILENO, "\n");
    // Don't quit here. It's just a notification.
}

ErrorSignalHandler::ErrorSignalHandler() {
    signal(SIGBUS, handler);
    signal(SIGFPE, handler);
    signal(SIGILL, handler);
    signal(SIGABRT, handler);
    signal(SIGSEGV, handler);
    signal(SIGSYS, handler);
    signal(SIGXCPU, handler);        // raised by SGE via s_vmem, see manpage queue_conf
    signal(SIGUSR1, notifyHandler);  // SIGUSR1/2 might be raised by SGE via -notify, see qsub
    signal(SIGUSR2, notifyHandler);
}

void ErrorSignalHandler::abort() {
    signal(SIGABRT, SIG_DFL);
    ::abort();
    signal(SIGABRT, handler);
}

static ErrorSignalHandler errorSignalHandler;

void abort() {
    errorSignalHandler.abort();
}

}  // namespace AssertionsPrivate
