from collections import namedtuple

Options = namedtuple(
    'SyntheticGHIProcessorOptions',
    []
)


class SyntheticGHIProcessor(object):
    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts

    def processData(self, data):
        return 3
