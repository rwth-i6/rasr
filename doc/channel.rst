Channel
=======

Channels are named output streams that can be configured and redirected individually. All system components use so called channels to produce output. Channels also take care of character set conversion and (optionally) XML formatting. They are used for logging and for storing small data structures in XML format. A Channel inherits from ``std::ostream`` so the usage is intuitive. The content written to a channel is sent to its targets. Channels are named output streams that can be configured and redirected individually. Channels are managed by a singleton ``Core::Channel::Manager`` that wraps ``rdbuf`` of ``std::cout`` and ``std::cerr``, which might cause problems under certain circumstances.

Channels cause NO (!) overhead compared to standard streams if
you use only a single destination stream.  Multiple destination
streams through redirection have the overhead of making multiple
copies of the data to be written (conversion to character strings
occurs only once!).

Upon creation a Channel determines its set of targets via the
standard configuration process.  E.g. if your component is
called ``"recognizer"`` then ``Channel(config, "statistics")``, will
look for a resource matching ``"recognizer.statistics.channel"``.
The resource's value is interpreted as a comma separated list
of target names, specifying were the channel's data is sent.


Channel Configuration
---------------------

Upon creation a Channel determines its set of targets via the standard configuration process. E.g. if your component is called recognizer and owns a channel named statistics, then the channel can be configured with a resource matching ``recognizer.statistics.channel``.

Channel Targets
---------------

The following channel targets are predefined:

stdout 
    standard output of process 
stderr 
    standard error output of process 
nil 
    suppress output 

All other target names cause the channel manager to create additional targets as needed. Each target is a configurable object registered under ``<application-title>.channels.<target-name>``. If the target name contains a dot, only the part after the dot is used as parameter name. By default a channel target will open a file by its own name and write all output to this file. This can be overridden with the file parameter.

Channel targets can be configured in several ways:

* The file name can be changed to something different from the target name using the ``"file"`` parameter.
* If the file already exists it is overwritten by default. By setting ``"append"`` to true, the channel manager will append to the file instead of overwriting.
* File output is buffered by default, which can cause long delay in the output.  Set ``"unbuffered"`` to change to line-buffering mode.
* Internally channels expect to be provided with UTF-8 encoded character data.  (Plain ASCII is a subset of UTF-8.)  It is  possible to specify a specific output encoding for each channel target, by setting the parameter ``"encoding"`` (ISO-8859-1 by default, set via ``Core::defaultEncoding`` in ``src/Core/Unicode.hh``).  The channel will convert the data into this character set encoding upon output.
* Channels feature automatic word-wrapping, which is disabled by default.  To enable word-wrapping, set the parameter ``"margin"`` to the number of characters per line.
* XML (and possibly other) text is automatically indented. The parameter "indentation" controls the depth of indentation.  Naturally, a value of zero, disable auto-indentation.
* zlib compression can be activated using the "compressed" parameter. The filename will be extended by the suffix ``".gz"`` if not already present

You can check whether a channel's output is actually used by calling ``isOpen()``.  Make use of this especially if your output needs additional calculations.

For plain text output use Channel.  If you want to produce XML output, use the derived class ``Core::XmlChannel``.


Target Parameters
-----------------

file 
    The file name can be changed to something different from the target name using the file parameter. 
append 
    If the file already exists it is overwritten by default. By setting append to true, the channel manager will append to the file instead of overwriting. 
unbuffered 
    File output is buffered by default, which can cause long delay in the output. Set unbuffered to change to line-buffering mode. 
encoding 
    Internally channels expect to be provided with UTF-8 encoded character data. (Plain ASCII is a subset of UTF-8.) It is possible to specify a specific output encoding for each channel target, by setting the parameter encoding (ISO-8859-1 by default). The channel will convert the data into this character set encoding upon output. 
margin 
    Channels feature automatic word-wrapping, which is disabled by default. To enable word-wrapping, set the parameter margin to the number of characters per line. 
indentation 
    XML (and possibly other) text is automatically indented. The parameter indentation controls the depth of indentation. Naturally, a value of zero, disables auto-indentation. 
compressed 
    zlib compression can be activated using the compressed parameter. The filename will be extended by the suffix ".gz" if not already present 

Example
-------

.. code-block :: ini

    [*]
    log.channel     = output-channel
    warning.channel = output-channel, stderr
    error.channel   = output-channel, stderr
    dot.channel     = nil

All channels named log of all components (because of the leading ``*``) are assigned to a target named output-channel.
The channels named warning and error (which are used for reporting of warning and error messages) will sent their output to the output-channel target and additionally to the standard error output.
The output of all dot channels is suppressed. 

Now we define the properties of the output-channel target:

.. code-block :: ini

    [*.channels.output-channel]
    file       = log/my-logfile.log
    append     = false
    encoding   = UTF-8
    unbuffered = true
    compressed = false

Thus, the content of output-channel will be written unbuffered and uncompressed to a file log/my-logfile.log using UTF-8 encoding. If the file already exists, it will be overwritten.

Default Channels
----------------

Most components write to the following channels:

error 
    error messages 
warning 
    warning messages 
log 
    log messages 

The Application provide the following channels:

system-info 
    information about the machine and the operating system 
version 
    software version information 
configuration 
    dump all configured parameters 
configuration-usage 
    dump used configurated parameters together with the requesting component 
time 
    run time 
