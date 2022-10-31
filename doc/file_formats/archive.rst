Archive
=======

An archive file is a container format (like tar or zip). It can contain arbitrary files or other types of named data. Archives can be stored as one single file (file archive) or as a directory (directory archive). The type of the archive is determined automatically by the archive option path.

Archives are used mainly for the caching of acoustic features, `FLF Lattice`, and alignment storage. The keys (or names) of the data streams (features, aligments, ...) in the archive are the segment names.

A set of archives can be combined to a single archive using the :ref:`Archiver Tool`.

**Configuration**

path 
    Pathname of the archive. If path is a directory or ends with a /, a directory archive is used. A file archive is used in all other cases. 
allow-overwrite 
    allow overwriting of existing files inside the archive (only for file archives) 


Bundle Archive
^^^^^^^^^^^^^^

Another kind of archive combination is a bundle archive. A bundle archives contains a list of archives: one path per line. The bundle archive must have the filename extension .bundle. It can be used like a normal archive. Bundle archives are read-only.
Example: feature.cache.bundle ::

    /work/user/my/experiment/data/features.cache.1
    /work/user/my/experiment/data/features.cache.2
    /work/user/my/experiment/data/features.cache.3

This creates a virtual combination of all 3 archives.

On first traversal, RASR will create a file ending in .bundle.idx.gz that contains a map from segment name to cache number to speed up further access.

Bundles are read-only archives.

Bundles support parameter ``*.max-open-files`` that allows to restrict the number of open file handles (in order to avoid exceeding `RLIMIT_NOFILE <http://man7.org/linux/man-pages/man2/getrlimit.2.html>`_). 
