import typing

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils import utils


def get_station_px_center(
        dataframe: pd.DataFrame,
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],):
    """
    A function that converts the coordinate values of a point on earth to that on an image

    Args:
        dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
        target_stations: A dictionary of all target stations with respective coordinates and elevation

    Returns:
        stations_px: A ``dict`` of station names as keys and their pixel coordinates as values [x: horizontal, y: vertical]
    """

    lats, lons = get_lats_longs_goes13_coords(dataframe)

    stations_pixels_coords = {}
    for name, coords in target_stations.items():
        # converts station's lats and lons to horizontal and vertical pixels
        x = np.argmin(np.abs(lons - coords[1]))
        y = np.argmin(np.abs(lats - coords[0]))
        stations_pixels_coords[name] = (x, y)

    return stations_pixels_coords

# auxiliary functions


def get_lats_longs_goes13_coords(catalog):
    goes13_ds = 'hdf5_16bit_path'
    lats = None
    lons = None

    all_paths = list(catalog.groupby(goes13_ds).groups.keys())
    i = 0
    while lats is None and lons is None and i < len(all_paths):
        with h5py.File(all_paths[i], "r") as h5_data:
            offset = np.random.randint(0, h5_data.attrs["global_dataframe_end_idx"] - h5_data.attrs["global_dataframe_start_idx"])
            lats, lons = utils.fetch_hdf5_sample("lat", h5_data, offset), utils.fetch_hdf5_sample("lon", h5_data, offset)
        i = i + 1

    assert lats is not None and lons is not None, 'No latitude and longitude values were found'
    return lats, lons


def show_stations(image, stations_px_coords):
    fig, ax = plt.subplots(1)

    # show the image
    ax.imshow(image, cmap='bone')

    if stations_px_coords:
        for name, coords in stations_px_coords.items():
            circle = patches.Circle(coords, 10, color='r')
            ax.add_patch(circle)
            ax.annotate(name, coords)

    plt.show()


def main():
    # expected usage
    app_config = {
        "dataframe_path": "/project/cq-training-1/project1/data/dummy_test_catalog.pkl",
        "goes13_dataset": 'hdf516',
        "_sample_image": '/project/cq-training-1/project1/data/hdf5v5_16bit/2014.07.11.0800.h5',
        "stations": {
            "BND": [40.05192, -88.37309, 230],
            "TBL": [40.12498, -105.23680, 1689],
            "DRA": [36.62373, -116.01947, 1007],
            "FPK": [48.30783, -105.10170, 634],
            "GWN": [34.25470, -89.87290, 98],
            "PSU": [40.72012, -77.93085, 376],
            "SXF": [43.73403, -96.62328, 473]
        }
    }

    dataframe_path = app_config['dataframe_path']
    catalog = pd.read_pickle(dataframe_path)
    stations = {k: (v[0], v[1], v[2]) for k, v in app_config['stations'].items()}

    stations_px_coordinates = get_station_px_center(catalog, stations)

    image = None
    with h5py.File(app_config['_sample_image'], "r") as h5_data:
        image = utils.fetch_hdf5_sample('ch2', h5_data, 35)
    show_stations(image, stations_px_coordinates)


if __name__ == '__main__':
    main()
