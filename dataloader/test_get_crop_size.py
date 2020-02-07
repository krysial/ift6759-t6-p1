import tensorflow as tf
import numpy as np
import functools

from get_crop_size import *


def return_dataloader():

    def gen(bs):

        for z in range(0, len(i), bs):
            yield i[z:z + bs], o[z:z + bs]

    i = np.ones((20, 3, 5, 7, 7))
    o = np.ones((20, 4))
    gene = functools.partial(gen, bs=11)
    dataloader = tf.data.Dataset.from_generator(gene, (tf.float32, tf.float32))
    return dataloader


data_loader = return_dataloader()
stations_px = {
    "BND": [40.01223, -88.46335],
    "TBL": [40.14563, -10.34521],
    "DRA": [36.22544, -11.34521],
    "FPK": [48.43425, -10.34521],
    "GWN": [34.52234, -89.34521],
    "PSU": [40.32324, -77.34521],
    "SXF": [43.23232, -96.34521]
}
