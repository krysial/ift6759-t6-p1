from collections import namedtuple
import pandas as pd
import numpy as np
from pvlib.location import Location

Options = namedtuple(
    'SyntheticGHIProcessorOptions',
    [
        'lat',
        'lon',
        'alt'
    ]
)


class SyntheticGHIProcessor(object):
    def __init__(self, opts: Options):
        super().__init__()
        self.opts = opts
        self.start_year = 2014
        self.setup_year_ghi()

    def setup_year_ghi(self):
        self.count = 0
        times = pd.date_range(
            start='%d-01-01' % (self.start_year),
            end='%d-01-01' % (self.start_year + 1),
            freq='15Min',
            tz='utc'
        )
        location = Location(
            self.opts.lat,
            self.opts.lon,
            altitude=self.opts.alt
        )
        self.cs = location.get_clearsky(times)

    def processData(self, data):
        if self.count < self.cs.shape[0]:
            ghi = self.cs['ghi'][self.count]
        else:
            self.start_year += 1
            self.setup_year_ghi()
            ghi = self.cs['ghi'][self.count]

        self.count += 1
        solar_power = np.random.rand()
        cloudness = 1 - (np.sum(data) / np.prod(data.shape))

        return ghi * cloudness
