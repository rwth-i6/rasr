__version__   = '$Revision: 1.3 $'
__date__      = '$Date: 2005/12/20 15:54:59 $'


import os, popen2, string

def execute(command, vars=None, capture=False, **kws):
    """
    Execute command in a sub-process.  The command string may contain
    shell-style variable substitutions (e.g. --foo $bar).  The values
    may be given as keyword arguments or in the vars dictionary.  If
    the argument capture is true, the standard output of the
    sub-process is redircted and the command's output is returned.  If
    command terminates with a non=zero exit code, an exceptio is
    raised.
    """
    mapping = dict()
    if vars:
	mapping.update(vars)
    mapping.update(kws)
    cmd = string.Template(command.replace('\n', ' ')).substitute(mapping)
    if capture:
	pipe = popen2.Popen3(cmd)
	pipe.tochild.close()
	result = pipe.fromchild.read()
	status = pipe.wait()
    else:
	result = None
	status = os.system(cmd)
    if status != 0:
	raise RuntimeError('command %r terminated with non-zero exit status %d' % (cmd, status))
    return result
