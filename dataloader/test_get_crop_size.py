import tensorflow as tf
import numpy as np
import functools

from get_crop_size import *

def return_dataloader():

    def gen(bs):

        for z in range(0,len(i),bs):
            yield i[z:z+bs], o[z:z+bs]

    i = np.ones((20,3,5,7,7))
    o = np.ones((20,4))
    gene = functools.partial(gen, bs=11)
    dataloader = 
tf.data.Dataset.from_generator(gene,(tf.float32,tf.float32))

    return dataloader

data_loader = return_dataloader()
stations_px = {
    "BND": (40, -88),
    "TBL": (40, -10),
    "DRA": (36, -11),
    "FPK": (48, -10),
    "GWN": (34, -89),
    "PSU": (40, -77),
    "SXF": (43, -96)
}


