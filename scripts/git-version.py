#!/usr/bin/env python3

import os

class ShellError(Exception):
	def __init__(self, res):
		self.exitCode = res
		assert self.exitCode != 0
		super(ShellError, self).__init__("exit code %i" % self.exitCode)

def sysexec(*args, **kwargs):
	import subprocess
	res = subprocess.call(args, shell=False, **kwargs)
	if res != 0: raise ShellError(res)

def sysexecVerbose(*args, **kwargs):
	print("sysexec: %s" % (args,))
	return sysexec(*args, **kwargs)

def sysexecOut(*args, **kwargs):
	from subprocess import Popen, PIPE
	kwargs.setdefault("shell", False)
	p = Popen(args, stdin=PIPE, stdout=PIPE, **kwargs)
	out, _ = p.communicate()
	if p.returncode != 0: raise ShellError(p.returncode)
	out = out.decode("utf-8")
	return out

def sysexecRetCode(*args, **kwargs):
	import subprocess
	res = subprocess.call(args, shell=False, **kwargs)
	valid = kwargs.get("valid", (0,1))
	if valid is not None:
		if res not in valid: raise ShellError(res)
	return res


def git_topLevelDir(gitdir=None):
	d = sysexecOut("git", "rev-parse", "--show-toplevel", cwd=gitdir).strip()
	assert(os.path.isdir(d))
	assert(os.path.isdir(d + "/.git"))
	return d

def git_headCommit(gitdir=None):
	return sysexecOut("git",  "rev-parse", "--short", "HEAD", cwd=gitdir).strip()

def git_commitRev(commit="HEAD", gitdir="."):
	if commit is None: commit = "HEAD"
	return sysexecOut("git", "rev-parse", "--short", commit, cwd=gitdir).strip()

def git_isDirty(gitdir="."):
	r = sysexecRetCode("git", "diff", "--no-ext-diff", "--quiet", "--exit-code", cwd=gitdir)
	if r == 0: return False
	if r == 1: return True
	assert(False)

def git_commitDate(commit="HEAD", gitdir="."):
	return sysexecOut("git", "show", "-s", "--format=%ci", commit, cwd=gitdir).strip()[:-6].replace(":", "").replace("-", "").replace(" ", ".")


if __name__ == "__main__":
	CurGitDir = git_topLevelDir(gitdir=os.path.dirname(__file__) or ".")
	CurGitCommit = git_headCommit(gitdir=CurGitDir)
	CurGitDirtyPostfix = "-dirty" if git_isDirty(gitdir=CurGitDir) else ""
	CurGitDate = git_commitDate(gitdir=CurGitDir)
	print(CurGitDate + "." + CurGitCommit + CurGitDirtyPostfix)
