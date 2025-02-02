import pytest
import typing
import numpy as np
import tensorflow as tf
import functools

from numpy.random import randint as r_i
from dataloader.get_crop_size import *


@pytest.fixture
def stations_px():
    stations = {
        "BND": [r_i(2, 99), r_i(2, 99)],
        "TBL": [r_i(2, 99), r_i(2, 99)],
        "DRA": [r_i(2, 99), r_i(2, 99)],
        "FPK": [r_i(2, 99), r_i(2, 99)],
        "GWN": [r_i(2, 99), r_i(2, 99)],
        "PSU": [r_i(2, 99), r_i(2, 99)],
        "SXF": [r_i(2, 99), r_i(2, 99)],
    }
    return stations


@pytest.fixture
def dataloader():
    def gen(bs):
        for z in range(0, len(i), bs):
            yield {'images': i[z:z + bs]}, o[z:z + bs]
    i = np.ones((200, 3, 5, 100, 100))
    o = np.ones((200, 4))
    gene = functools.partial(gen, bs=11)
    dataloader = tf.data.Dataset.from_generator(gene, ({
        'images': tf.float32,
    }, tf.float32))
    return dataloader


def test_output(dataloader, stations_px):
    sq_crop_side_len = get_crop_size(stations_px, dataloader)
    assert (sq_crop_side_len is not None) and isinstance(sq_crop_side_len, int)


def test_zero_check(dataloader, stations_px):
    sq_crop_side_len = get_crop_size(stations_px, dataloader)
    assert sq_crop_side_len != 0


def test_positive_check(dataloader, stations_px):
    sq_crop_side_len = get_crop_size(stations_px, dataloader)
    assert sq_crop_side_len > 0


def test_frame_limit_raw_image(dataloader, stations_px):
    sq_crop_side_len = get_crop_size(stations_px, dataloader)
    for data, target in dataloader:
        break
    L, B = data['images'].shape[-2:]
    assert (sq_crop_side_len < L) and (sq_crop_side_len < B)
