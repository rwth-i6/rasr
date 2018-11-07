"""
Stepper: A light-weight framework for writing long-running multi-step
script jobs.

How to use it:
 * Derive from Stepper.
 * Define methodes names "step_1", "step_2", ...
   Use doc-strings to describe what is done in the step.
   Of course you can define and call methodes with different names.
 * Call run().
   Stepper will run each step function in order.

What Stepper will do for you:
 * You can start and stop at any step.
 * All class member variabes are stored persistently.
 * Stepper will automatically resume from the next incomplete step.

What we want to add in the future:
 * integration with queueing systems
 * "array" steps for parallel processing
"""


__version__   = '$Revision: 1.11 $'
__date__      = '$Date: 2005/12/15 15:38:50 $'


# ===========================================================================
# This part should become an independent module

import os, shelve
import cPickle as pickle


class Stepper(object):
    needToSave_ = dict()

    def __init__(self, store, name=None, options=None):
	self.needToSave_ = dict((key, True) for key in self.__dict__)
	self.store_ = store
	self.name = name
	if name:
	    self.prefix_ = name + '.'
	else:
	    self.prefix_ = ''
	self.options_ = options
	self.shelve_ = shelve.open(self.store_, protocol=pickle.HIGHEST_PROTOCOL)
	self.currentStep_ = 1

    def __getattr__(self, key):
	if key in self.__dict__:
	    return self.__dict__[key]
	shkey = self.prefix_ + key
	if shkey in self.shelve_:
	    print 'restoring', key
	    self.__dict__[key] = self.shelve_[shkey]
	    self.needToSave_[key] = False
	    return self.__dict__[key]
	raise AttributeError('%s instance has no attribute %r' % (self.__class__.__name__, key))

    def __setattr__(self, key, value):
	self.__dict__[key] = value
	self.needToSave_[key] = True

    def save(self):
	for key, value in self.__dict__.iteritems():
	    if not key.endswith('_') and self.needToSave_[key]:
		print 'saving', key
		self.shelve_[self.prefix_ + key] = value
		self.needToSave_[key] = False
	self.shelve_[self.prefix_ + 'currentStep_'] = self.currentStep_
	self.shelve_.sync()

    def run(self, fromStep=None, toStep=None):
	if fromStep is None:
	    fromStep = self.options_.start
	if toStep is None:
	    toStep = self.options_.stop
	if fromStep is None:
	    if (self.prefix_ + 'currentStep_') in self.shelve_:
		self.currentStep_ = self.shelve_[self.prefix_ + 'currentStep_']
	    else:
		self.currentStep_ = 1
	else:
	    self.currentStep_ = fromStep
	while (toStep is None) or (self.currentStep_ <= toStep):
	    try:
		step = getattr(self, 'step_%d' % self.currentStep_)
	    except AttributeError:
		break
	    print 'executing step %d: %s ...' % (self.currentStep_, step.__doc__.strip())
	    step()
	    self.currentStep_ += 1
	    self.save()
	    print

    @classmethod
    def addOptions(cls, optparser):
	optparser.add_option(
	    '--start', type='int',
	    help='(re-) start execution with step N)', metavar='N')
	optparser.add_option(
	    '--stop', type='int',
	    help='stop execution after step N)', metavar='N')
