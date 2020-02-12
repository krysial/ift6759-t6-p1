from collections import namedtuple
import pandas as pd
import numpy as np
from pvlib.location import Location

Options = namedtuple(
    'SyntheticGHIProcessorOptions',
    [
        'lat',
        'lon',
        'alt',
        'offsets'
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

    def processData(self, _data):
        results = []
        max_offset = self.opts.offsets[-1]
        solar_power = np.random.randint(70, high=100) / 100

        for offset in self.opts.offsets:
            data = _data[:, max_offset - offset, :, :]

            if self.count + max_offset < self.cs.shape[0]:
                ghi = self.cs['ghi'][self.count + max_offset - offset]
            else:
                self.start_year += 1
                self.setup_year_ghi()
                ghi = self.cs['ghi'][self.count + max_offset - offset]

            cloudness = np.average(
                solar_power * (1 - (np.sum(data, axis=0) / data.shape[0]))
            )

            results.append(ghi * cloudness)

        self.count += 1

        return results
