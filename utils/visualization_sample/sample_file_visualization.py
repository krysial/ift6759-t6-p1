import json
import os
import sys
sys.path.append('../')
import utils  # noqa

DATA_FOLDER = '/project/cq-training-1/project1/data/'
CATALOG_PATH = os.path.join(
    DATA_FOLDER, 'catalog.helios.public.20100101-20160101.pkl')
EVAL_PATH = 'dummy_test_cfg.json'
H5_PATH = os.path.join(DATA_FOLDER, 'hdf5v5_16bit/2014.07.11.0800.h5')

with open(EVAL_PATH) as f:
    eval_conf = json.load(f)

print('Start visualization')
utils.viz_hdf5_imagery(
    H5_PATH, ['ch1', 'ch2', 'ch3', 'ch4', 'ch6'], CATALOG_PATH, eval_conf['stations'])
