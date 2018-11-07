#!/usr/bin/env python

# call Sprint via:
# nn-trainer --action=python-control --pymod-name=python_control_demo

# also try these additional options:
# --python-control-loop-type=iterate-corpus --*.extract-features=false --*.corpus.file=/u/tuske/work/ASR/switchboard/corpus/xml/train.corpus.gz --*.corpus.partition=100000

import sys
import os
from subprocess import Popen, check_call

if sys.platform == "darwin":
	ExecPath = "./nn-trainer.darwin-x86_64-standard"
else:
	ExecPath = "./nn-trainer.linux-x86_64-standard"

def init(**kwargs):
	print("init: %r" % kwargs)
	return Control()

class Control:
	def run_control_loop(self, callback, **kwargs):
		print("run_control_loop: %r, %r" % (callback, kwargs))
		print(">> Sprint Version: %r" % callback("version"))
		shell_ns = {"callback": callback}
		shell_ns.update(kwargs)
		print(">> Use callback(cmd, ...).")
		debug_shell(shell_ns, shell_ns)

	def init_processing(self, **kwargs):
		print("init_processing: %r" % kwargs)

	def process_segment(self, **kwargs):
		print("process_segment: %r" % kwargs)

	def exit(self, **kwargs):
		print("exit: %r" % kwargs)


def simple_debug_shell(globals, locals):
	try: import readline
	except ImportError: pass # ignore
	COMPILE_STRING_FN = "<simple_debug_shell input>"
	while True:
		try:
			s = raw_input("> ")
		except (KeyboardInterrupt, EOFError):
			print("breaked debug shell: " + sys.exc_info()[0].__name__)
			break
		if s.strip() == "": continue
		try:
			c = compile(s, COMPILE_STRING_FN, "single")
		except Exception as e:
			print("%s : %s in %r" % (e.__class__.__name__, str(e), s))
		else:
			set_linecache(COMPILE_STRING_FN, s)
			try:
				ret = eval(c, globals, locals)
			except (KeyboardInterrupt, SystemExit):
				print("debug shell exit: " + sys.exc_info()[0].__name__)
				break
			except Exception:
				print("Error executing %r" % s)
				better_exchook(*sys.exc_info(), autodebugshell=False)
			else:
				try:
					if ret is not None: print(ret)
				except Exception:
					print("Error printing return value of %r" % s)
					better_exchook(*sys.exc_info(), autodebugshell=False)

def debug_shell(user_ns, user_global_ns):
	ipshell = None
	if not ipshell:
		try:
			import IPython
			import IPython.terminal.embed
			class DummyMod(object): pass
			module = DummyMod()
			module.__dict__ = user_global_ns
			module.__name__ = "DummyMod"
			ipshell = IPython.terminal.embed.InteractiveShellEmbed(
				user_ns=user_ns, user_module=module)
		except Exception:
			pass
	if ipshell:
		ipshell()
	else:
		simple_debug_shell(user_global_ns, user_ns)


if __name__ == "__main__":
	mydir = os.path.dirname(__file__)
	os.chdir(mydir)
	assert os.path.exists(ExecPath), "%s/%s not found" % (mydir, ExecPath)
	args = [ExecPath, "--*.action=python-control", "--pymod-name=python_control_demo"]
	print("Call: %r" % args)
	check_call(args)

