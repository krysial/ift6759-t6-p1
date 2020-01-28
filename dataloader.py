import pandas as pd
import numpy as np
import tensorflow as tf
import tqdm
import datetime
import typing


def prepare_dataloader(
        dataframe: pd.DataFrame,
        target_datetimes: typing.List[datetime.datetime],
        station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
        target_time_offsets: typing.List[datetime.timedelta],
        config: typing.Dict[typing.AnyStr, typing.Any],
        target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],) -> tf.data.Dataset:

"""
A function to prepare & return tensorflow dataloader
"""

   def create_data_generator(
           dataframe: pd.DataFrame,
           target_datetimes: typing.List[datetime.datetime],
           station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
           target_time_offsets: typing.List[datetime.timedelta],
           config: typing.Dict[typing.AnyStr, typing.Any],):

   """
   A function to create a generator to yield data to the dataloader
   """

      def get_GHI_targets(
              dataframe: pd.DataFrame,
              batch_of_datetimes: typing.List[datetime.datetime],
              station: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],
              target_time_offsets: typing.List[datetime.timedelta],
              config: typing.Dict[typing.AnyStr, typing.Any],) -> np.ndarray:

      """
      A function to get the sequence of target GHI values

      Args:
          dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
          batch_of_datetimes: a batch of timestamps that is required by the data loader to provide targets for the model.
          station: a map of station names of interest paired with their coordinates (latitude, longitude, elevation).
          target_time_offsets: the list of timedeltas to predict GHIs for (by definition: [T=0, T+1h, T+3h, T+6h]).
          config: configuration dictionary holding any extra parameters that might be required for tuning purposes.

      Returns:
          A numpy array of shape (batch_size, 4)
      """

         assert targets.shape == (len(batch_of_datetimes),4)
         return targets


      def get_raw_images(
              dataframe: pd.DataFrame,
              batch_of_datetimes: typing.List[datetime.datetime],
              config: typing.Dict[typing.AnyStr, typing.Any],) -> np.ndarray:

      """
      A function to get the raw input images as taken by the satelite

      Args:
          dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
          batch_of_datetimes: a batch of timestamps that is required by the data loader to provide raw uncropped versions of
                              the satelite images for the model.
          config: configuration dictionary holding any extra parameters that might be required for tuning purposes.

      Returns:
          A numpy array of shape (batch_size, temporal_seq, channels, length, bredth)
      """

         assert images.shape==(len(batch_of_datetimes),config['no_of_temporal_seq'],5,L,B)
         return images


      for i in range(0, len(target_datetimes), batch_size):
         batch_of_datetimes = target_datetimes[i:i+batch_size]
         targets = get_GHI_targets(dataframe, batch_of_datetimes, station, target_time_offsets, config)
         images = get_raw_images(dataframe, batch_of_datetimes, config)
         yield images, targets


   def get_crop_frame_size(
           dataframe: pd.DataFrame,
           target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],) -> int:

   """
   A function to return an ``int`` value corresponding to the crop frame side length
   """

      def get_station_px_center(
           dataframe: pd.DataFrame,
           target_stations: typing.Dict[typing.AnyStr, typing.Tuple[float, float, float]],):

      """
      A function that converts the coordinate values of a point on earth to that on an image

      Args:
          dataframe: a pandas dataframe that provides the netCDF file path (or HDF5 file path and offset).
          target_stations: A dictionary of all target stations with respective coordinates and elevation 

      Returns:
          stations_px: A ``dict`` of station names as keys and their pixel coordinates as values
          L: An ``int`` value of length dimension of image
          B: An ``int`` value of bredth dimension of image
      """

         return stations_px, L, B


      def crop_size(
           stations_px: typing.Dict[typing.AnyStr, typing.Tuple[float, float]],
           L: int, B: int,) -> int:

      """
      A function to find the best/smallest square crop frame size. It finds a crop size such that 
      there are no overlaps and is a valid crop for all stations.

      Args:
          stations: a map of station names with their pixel coordinates on image.
          L: length dimension of image
          B: bredth dimension of image

      Returns:
          An ``int`` value which can be used to define the square crop frame side length.
      """

         return sq_crop_side_len


      stations_px, L, B = get_station_px_center(dataframe, target_stations)
      crop_frame_size = crop_size(stations_px, L, B)
      return crop_frame_size


   if config['crop_size'] == None:
      config['crop_size'] = get_crop_frame_size(dataframe, target_stations)

   generator = create_data_generator(dataframe, target_datetimes, station, target_time_offsets, config)
   data_loader = tf.data.Dataset.from_generator(generator)

   return data_loader
