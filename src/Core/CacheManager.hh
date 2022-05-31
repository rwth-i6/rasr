#ifndef _CORE_CACHE_MANAGER
#define _CORE_CACHE_MANAGER

#include <Modules.hh>

#ifdef MODULE_CORE_CACHE_MANAGER

#include <string>

namespace Core {

/**
 * Resolves cache-manager commands. Cache manager commands are
 * enclosed by "`cf`" and "`".
 *
 * If the "-d" flag is given the original path and the one given
 * by the cache manager are stored. By calling
 * @copyLocalCacheFiles() the local files are copied to the
 * original path. This is useful for files generated by sprint
 * as there will be only one write to the work file server.
 */
std::string resolveCacheManagerCommands(std::string const&);

/**
 * Copies all files that were passed through the cache-manager
 * with the "-d" flag to their final location.
 */
void copyLocalCacheFiles();

}  // namespace Core

#endif  // MODULE_CORE_CACHE_MANAGER

#endif  // _CORE_CACHE_MANAGER
