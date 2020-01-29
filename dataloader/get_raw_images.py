

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

    assert images.shape == (len(batch_of_datetimes),
                            config['no_of_temporal_seq'], 5, L, B)
    return images
