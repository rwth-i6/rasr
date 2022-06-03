#include "CacheManager.hh"

#ifdef MODULE_CORE_CACHE_MANAGER

#include <algorithm>
#include <cstring>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#include "Assertions.hh"

// if the cache manager is for some reason at a different location
// overwrite this macro with a compile flag
#ifndef CACHE_MANAGER_PATH
// clang-format off
#define CACHE_MANAGER_PATH /usr/local/cache-manager/cf
// clang-format on
#endif

// note that in this macro X is not expanded before stringification
#define STRINGIFY2(X) #X
// thus we use this macro to do the expansion
#define STRINGIFY(X) STRINGIFY2(X)

namespace {

/* This functions calls the cache-manager with the given args and returns
 * its output in a string. The last newline is truncated to make it consistent
 * with the behaviour of a shell
 */
std::string run_cache_manager(std::vector<std::string> const& args) {
    // create a pipe to communicate with the child process
    int pipe_fd[2u];
    if (pipe2(pipe_fd, O_CLOEXEC) == -1) {
        std::stringstream error;
        error << "Error creating unnamed pipe: " << strerror(errno);
        throw std::runtime_error(error.str());
    }

    pid_t pid = fork();
    if (pid == -1) {
        std::stringstream error;
        error << "Error while forking: " << strerror(errno);
        throw std::runtime_error(error.str());
    }
    else if (pid == 0) {
        // child

        errno = 0;
        // duplicate the pipe write end to the output of the child process
        // loop until not interruted by signal
        while (dup2(pipe_fd[1u], STDOUT_FILENO) == -1 and errno == EINTR) {
        }
        if (errno != 0 and errno != EINTR) {
            std::stringstream error;
            error << "Error while duplicating fd: " << strerror(errno);
            throw std::runtime_error(error.str());
        }

        // convert args to array of mutable c-strings
        std::vector<char*> argv(args.size() + 2u);
        argv[0u] = new char[3u];
        std::memcpy(argv[0u], "cf", 3u);
        for (size_t idx = 0u; idx < args.size(); idx++) {
            argv[idx + 1u] = new char[args[idx].size() + 1u];
            std::memcpy(argv[idx + 1u], args[idx].c_str(), args[idx].size() + 1u);
        }
        // needed by execv to determine the end of the argument array
        argv[argv.size() - 1u] = NULL;

        execv(STRINGIFY(CACHE_MANAGER_PATH), const_cast<char* const*>(&argv[0u]));
    }

    // parent
    // close the write end of the pipe, the parent does not need it
    close(pipe_fd[1u]);

    std::stringstream ss;
    ssize_t           bytes_read;
    char              buffer[1024u];
    while ((bytes_read = read(pipe_fd[0u], buffer, sizeof(buffer))) > 0) {
        ss << std::string(buffer, bytes_read);
    }
    if (bytes_read < 0) {
        std::stringstream error;
        error << "Error reading from child process: " << strerror(errno);
        throw std::runtime_error(error.str());
    }

    // truncate last newline
    std::string cmd_output = ss.str();
    size_t      trim_end   = cmd_output.find_last_not_of("\n");
    if (trim_end != std::string::npos) {
        cmd_output = cmd_output.substr(0u, trim_end + 1u);
    }

    // wait for child to exit, if wait is not called the child will become a zombie
    waitpid(pid, NULL, 0);
    close(pipe_fd[0u]);

    return cmd_output;
}

std::vector<std::string> sources;
std::vector<std::string> destinations;

}  // namespace

std::string Core::resolveCacheManagerCommands(std::string const& value) {
    std::string result;
    size_t      begin, pos = 0u;

    while ((begin = value.find("`cf ", pos)) != std::string::npos) {
        result += value.substr(pos, begin - pos);
        pos        = begin + 4u;
        begin      = pos;
        size_t end = value.find("`", pos);
        if (end != std::string::npos) {
            std::string params = value.substr(begin, end - begin);

            if (params.size() > 0u) {
                std::stringstream                  ss(params);
                std::istream_iterator<std::string> start(ss);
                std::istream_iterator<std::string> end;
                std::vector<std::string>           tokens(start, end);

                if (tokens.size() == 0) {
                    throw std::runtime_error("No parameters given for cache manager call");
                }

                std::string out = run_cache_manager(tokens);

                bool copy_at_exit = std::find(tokens.begin(), tokens.end(), std::string("-d")) != tokens.end();
                if (copy_at_exit) {
                    sources.push_back(out);
                    destinations.push_back(tokens[tokens.size() - 1u]);
                }

                result += out;
            }

            pos = end + 1u;
        }
        else {
            std::stringstream error_stream;
            error_stream << "configuration error: unclosed back-tick in value \"" << value << '"';
            throw std::runtime_error(error_stream.str());
        }
    }
    result += value.substr(pos, std::string::npos);

    return result;
}

void Core::copyLocalCacheFiles() {
    require(sources.size() == destinations.size());
    for (size_t idx = 0u; idx < sources.size(); idx++) {
        std::vector<std::string> args;
        args.push_back("-cp");
        args.push_back(sources[idx]);
        args.push_back(destinations[idx]);
        run_cache_manager(args);
    }
}

#endif  // MODULE_CORE_CACHE_MANAGER
