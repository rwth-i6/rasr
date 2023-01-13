Archiver Tool
=============

The archiver tool manipulates archive files.
Unlike the other tools, the archiver does not use configuration files, but simple command line options.

Usage
-----

 ::

    archiver [OPTION] <archive> <FILE>...

**Options**:

| ``--compress <bool>`` : compress new files added to the archive
| ``--mode <mode>`` : choose operational mode
| ``--verbose <bool>`` : be a bit more verbose
| ``--quiet <bool>`` : less output
| ``--select <file>`` : apply operation only to files (inside the archive) listed in <file>; only supported by modes combine and copy
| ``--overwrite <mode>`` : what to do when archive member already exists
| ``--type <T>`` : interpret cache as an object of type T (align, feat, ascii, bin-matrix, flow-cache) when --mode=show
| ``--allophone-file <a.txt>`` : use this allophone file to resolve symbols when calling with --mode show --type align

**Modes**:

| add          : add files or directories to archive
| combine      : combine other archives into new one
| copy         : copy files between archives directly (option 'compress' is ignored)
| extract      : extract single files with path
| extractAll   : extract all files to given directory
| list         : list archive(s) (default)
| remove       : remove single files from archive
| recover      : recover archive (if internal structure is broken)
| show         : serialize and print file content to stdout, if possible (requires --type)

**Overwrite-Modes**:

| no           : no overwriting
| yes          : overwrite files with the same name
| check        : check for data equality of archive members with the same name


Examples
--------

Combine archives ``features.1.cache`` and ``features.2.cache`` to a compressed file archive ``features.cache``::

    archiver \
      --mode=combine   \
      --compress=true  \
      features.cache   \
      features.1.cache \
      features.2.cache

Extract files "EPPS/20060704_0905_1205_OR_SAT/10603.5-10633.7" and "EPPS/20060704_0905_1205_OR_SAT/10560.3-10565.45" from archive ``eval07en.feature-cache``::

    archiver \
      --mode=extract \
      eval07en.feature-cache \
      EPPS/20060704_0905_1205_OR_SAT/10603.5-10633.7 \
      EPPS/20060704_0905_1205_OR_SAT/10560.3-10565.45

Create a subset of a cache given a file list.
Please note: If you use a list of segments with full names, remember to copy ``<segment-full-name>.attribs`` as well. ::

    archiver \
      --mode=copy    \
      --select=filelist.txt \
     subset.cache \
     full.cache

