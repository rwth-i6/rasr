"""
Analog Plug-in for processing time (esp. real time factor)
"""

__version__   = '$Revision$'
__date__      = '$Date$'


from analog_util.analog import Collector, Field


class RealTime(Collector):
    id     = 'time'
    name   = 'time'
    fields = [Field('duration', 7, '%7.1f', 's'),
              Field('CPU',      7, '%7.1f', 's'),
              Field('rtf',      6, '%6.2f') ]

    def __call__(self, data):
        cpuTime  = sum(data['user time'])
        duration = sum(data['real time'])
        return list(zip(self.fields, [
            duration, cpuTime, cpuTime / duration if duration > 0.0 else 0.0 ]))
