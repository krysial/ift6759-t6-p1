import argparse
import datetime
import json
import os
import typing

import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm


def rescale_GHI(target_GHI):
    # Rescaling the normalised station_GHI predictions
    # station_GHI were normalized based on quantiles
    median_station_GHI = 297.487143
    quantile_diff_station_GHI = 470.059048
    target_GHI_ = (target_GHI * quantile_diff_station_GHI) + median_station_GHI
    assert target_GHI_.shape == target_GHI.shape
    assert target_GHI_.ndim == target_GHI.ndim
    return target_GHI_
