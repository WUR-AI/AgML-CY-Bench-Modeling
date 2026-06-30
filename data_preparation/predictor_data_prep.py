import os, time
import numpy as np
import rasterio
import rasterio.features
import pandas as pd
import geopandas as gpd
import multiprocessing as mp
from itertools import repeat
import logging
import argparse
import warnings

from cybench.config import DATASETS, PATH_DATA_DIR, PATH_POLYGONS_DIR, REPO_DIR
from cybench.util.geo import get_shapes_from_polygons

warnings.filterwarnings("ignore")


def init_worker_logging():
    """Disable file logging in worker processes to avoid RotatingFileHandler races."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(logging.NullHandler())


log = logging.getLogger(__name__)


EU_COUNTRY_CODE_KEY = "CNTR_CODE"
EU_ADMIN_LEVEL_KEY = "LEVL_CODE"

# Country codes to admin level
# Austria (AT), Belgium (BE), Bulgaria (BG), Czech Republic (CZ), Germany (DE), Denmark (DK),
# Estonia (EE), Greece (EL), Spain (ES), Finland (FI), France (FR), Croatia (HR), Hungary (HU),
# Ireland (IE), Italy (IT), Lithuania (LT), Latvia (LV), The Netherlands (NL), Poland (PL),
# Portugal (PT), Romania (RO), Sweden (SE), Slovakia (SK)
EU_COUNTRIES = {
    "AT": 2,
    "BE": 2,
    "BG": 2,
    "CZ": 3,
    "DE": 3,
    "DK": 3,
    "EE": 3,
    "EL": 3,
    "ES": 3,
    "FI": 3,
    "FR": 3,
    "HR": 2,
    "HU": 3,
    "IE": 2,
    "IT": 3,
    "LT": 3,
    "LV": 3,
    "NL": 2,
    "PL": 2,
    "PT": 2,
    "RO": 3,
    "SE": 3,
    "SK": 3,
}

# Angola (AO), Burkina Faso (BF), Ethiopia (ET), Lesotho (LS), Madagascar (MG), Malawi (MW),
# Mozambique (MZ), Niger (NE), Senegal (SN), Chad (TD), South Africa (ZA), Zambia (ZM)
FEWSNET_COUNTRIES = [
    "AO",
    "BF",
    "ET",
    "LS",
    "MG",
    "MW",
    "MZ",
    "NE",
    "SN",
    "TD",
    "ZA",
    "ZM",
]
FEWSNET_ADMIN_ID_KEY = "adm_id"


##################
# Directory paths
##################

AGML_ROOT = r"/lustre/backup/SHARED/AIN/agml"
DATA_DIR = os.path.join(AGML_ROOT, "predictors")
OUTPUT_DIR = os.path.join(AGML_ROOT, "python-output")

#####################
# Start and end years
#####################
# NOTES:
# FPAR data starts from 2001. There may be some data from 2000, but it's not complete.
# GLDAS data starts from 2003. 2003 is not complete.
START_YEAR = 2001
END_YEAR = 2023

########
# crop
########
CROPS = ["wheat", "maize"]

#####################################################
# predictors
# NOTE: key should match directory name containing raster files
#####################################################

ALL_INDICATORS = {
    "prec": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "tmax": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "tmin": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "tavg": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "rad": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "ndvi": {
        "source": "MOD09CMG",
        "is_time_series": True,
        "is_categorical": False,
    },
    "fpar": {
        "source": "JRC_FPAR500m",
        "is_time_series": True,
        "is_categorical": False,
    },
    "vpd": {
        "source": "AgERA5",
        "is_time_series" : True,
        "is_categorical": False,
    },
    "et0": {
        "source": "AgERA5",
        "is_time_series": True,
        "is_categorical": False,
    },
    "rsm": {
        "source": "GLDAS",
        "is_time_series": True,
        "is_categorical": False,
    },
    "ssm": {
        "source": "GLDAS",
        "is_time_series": True,
        "is_categorical": False,
    },
    "awc": {
        "source": "WISE_Soil",
        "is_time_series": False,
        "is_categorical": False,
    },
    "bulk_density": {
        "source": "WISE_Soil",
        "is_time_series": False,
        "is_categorical": False,
    },
    "drainage_class": {
        "source": "WISE_Soil",
        "is_time_series": False,
        "is_categorical": True,
    },
    "sos": {
        "source": "ESA_WC_Crop_Calendars",
        "is_time_series": False,
        "is_categorical": False,
    },
    "eos": {
        "source": "ESA_WC_Crop_Calendars",
        "is_time_series": False,
        "is_categorical": False,
    },
    "latitude": {
        "source": "Location",
        "is_time_series": False,
        "is_categorical": False,
    },
    "longitude": {
        "source": "Location",
        "is_time_series": False,
        "is_categorical": False,
    },
    "rainfed_temperate_cereals_yield": {
        "source": "LPJmL",
        "is_time_series": True,
        "is_categorical": False,
    },
    "rainfed_maize_yield": {
        "source": "LPJmL",
        "is_time_series": True,
        "is_categorical": False,
    },
    "irrigated_temperate_cereals_yield": {
        "source": "LPJmL",
        "is_time_series": True,
        "is_categorical": False,
    },
    "irrigated_maize_yield": {
        "source": "LPJmL",
        "is_time_series": True,
        "is_categorical": False,
    },
}

# used to read .nc files
AGERA5_VARIABLES = {
    "tmax": "Temperature_Air_2m_Max_24h",
    "tmin": "Temperature_Air_2m_Min_24h",
    "tavg": "Temperature_Air_2m_Mean_24h",
    "prec": "Precipitation_Flux",
    "rad": "Solar_Radiation_Flux",
    "et0": "ReferenceET_PenmanMonteith_FAO56",
    "vpd": "Vapour_Pressure_Deficit_at_Maximum_Temperature",
}

LPJML_VARIABLES = {
    "rainfed_temperate_cereals_yield": "harvestc",
    "rainfed_maize_yield": "harvestc",
    "irrigated_temperate_cereals_yield": "harvestc",
    "irrigated_maize_yield": "harvestc",
}


"""
@author: Joint Research Centre - D5 Food Security - ASAP
"""
SUPPRESS_ERRORS = True


class UnableToExtractStats(Exception):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    """

    pass


def read_masked(
    ds,
    mask,
    indexes=None,
    window=None,
    use_pixels="CENTER",
    out_mask_path=None,
    *args,
    **kwargs
):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Reads the data fom the raster file and returns it as a numpy masked array.
    The geometry is used to define the mask.
    It returns only a subset of the data.
        It reads only the raster window containing the geometry. This optimizes the performance by reducing disk reads.
    The "use_pixels" parameter can be used to define how to handle the border pixels.
        Some require shapely to be available on the system.

    :param ds: Raster file path or an instance of a rasterio.DatasetReader (already opened raster file).
    :param mask: iterable over geometries used to mask the array, mask is a read mask
                 (only data covered by the geometry is used)
    :param indexes: (list of ints or a single int, optional)
                    – If indexes is a list, the result is a 3D array,
                    but is a 2D array if it is a band index number.
    :param window: Optional window param to extract a partial dataset.
    :param use_pixels: Parameter that defines what pixels to use when masking with input geometry.
        CONTAINED - Only use pixels that are fully contained within the request geometry (requires shapely)
        ALL - Use all pixels touched by request geometry
        CENTER - Use all pixels whose center is within the request geometry
    :param out_mask_path: path on disk where to save the mask as geoTiff, skips if set to None
    :return: numpy array, optionally saves the mask as geoTiff to specified location
    """
    # open file
    dataset = ds if isinstance(ds, rasterio.DatasetReader) else rasterio.open(ds)

    # determine masking
    if use_pixels.upper() == "CONTAINED":
        mask_invert = True
        mask_all_touched = True
    elif use_pixels.upper() == "ALL":
        mask_invert = False
        mask_all_touched = True
    elif use_pixels.upper() == "CENTER":
        mask_invert = False
        mask_all_touched = False

    # get read window
    if window:
        _window = window
    else:
        _window = rasterio.features.geometry_window(dataset, mask)

    # read the source dataset
    source = dataset.read(indexes, window=_window, *args, **kwargs)
    # return empty source if the window missed the raster
    if 0 in source.shape:
        return source

    # create invert mask for CONTAINED (requires Shapely)
    if mask_invert:
        import shapely.geometry

        # create geometry difference with intersect geom
        dataset_window_bounds = rasterio.windows.bounds(_window, dataset.transform)
        mask_shapely_geom = shapely.geometry.box(
            *list(dataset_window_bounds)
        ).difference(shapely.geometry.shape(mask))
        mask = (
            None
            if mask_shapely_geom.is_empty
            else shapely.geometry.mapping(mask_shapely_geom)
        )

    # create transform and shape
    out_shape = source.shape[-2:]
    if window:
        # define transform with custom window and out shape
        out_transform = rasterio.transform.from_bounds(
            *dataset.window_bounds(window), *reversed(out_shape)
        )
    else:
        out_transform = dataset.window_transform(_window)

    # create the mask array matching the raster windowed reading
    input_geom_mask = rasterio.features.geometry_mask(
        mask,
        transform=out_transform,
        invert=mask_invert,
        out_shape=out_shape,
        all_touched=mask_all_touched,
    )
    # write the mask output
    if out_mask_path:
        with rasterio.open(
            out_mask_path,
            "w",
            driver="Gtiff",
            height=input_geom_mask.shape[0],
            width=input_geom_mask.shape[1],
            count=1,
            dtype=np.uint8,
            crs=dataset.crs,
            transform=out_transform,
        ) as tmp_dataset:
            tmp_dataset.write(input_geom_mask.astype(np.uint8), 1)

    # mask data arrays
    source = np.ma.array(source, mask=input_geom_mask)

    return source


def round_window(window):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Outputs a copy of the window rounded to the nearest whole pixel.
    Rounds the absolute size of the window extent which makes sure that all pixel cneters covered by the input window
    are included.

    :param window: Input window
    :return: rasterio.windows.Window
    """
    _col_off = round(window.col_off)
    _row_off = round(window.row_off)
    _width = round(window.col_off + window.width) - _col_off
    _height = round(window.row_off + window.height) - _row_off
    return rasterio.windows.Window(_col_off, _row_off, _width, _height)


def get_common_bounds_and_shape(geom, ds_list):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Returns a unified read window, bounds and resolution, for a given geometry on all raster datasets in a list.
    Used when combining datasets with heterogeneous resolution to read data with the extent (bounds) of the lowest
    resolution dataset and to use the resolution (shape) of the highest resolution dataset.
    Useful when trying to get unified arrays to perform calculation from heterogeneous datasets.
    The read window and resolution is defined with a following process.
        1. we find the lowest and highest resolution datasets by comparing their geometry window sizes (all pixels
           touched by geom).
        2. we get the geometry window bounds of the lowest resolution dataset and align them to the highest
           resolution dataset.
            - lowest resolution dataset window should be the maximum extent, extents covering all other dataset
              windows.
            - We align the bounds to the highest resolution dataset by defining the bounds window on the hres
              dataset and rounding them to the closest whole pixel
        3. we define unified read window and out_shape
            - We use the aligned bounds to define a read window for all datasets
            - We use the resolution of the hres rounded bounds window shape as the read out_shape

    :param geom: GeoJSON-like feature (implements __geo_interface__) – feature collection, or iterable over geometry.
    :param ds_list: List of raster file paths or rasterio datasets
    :return: Pair of tuples, shape and bounds - ((x_min, y_min, x_max, y_max), (rows, columns))
    """
    max_res_ds = None
    max_res_window_size = None
    min_res_window_size = None
    max_extent_window_bounds = None

    for ds in ds_list:
        dataset = ds if isinstance(ds, rasterio.DatasetReader) else rasterio.open(ds)
        # dataset = ds.to_rio()
        ds_window = rasterio.features.geometry_window(dataset, geom)

        if not max_res_ds:  # just assign values to all vars if first file in the loop
            max_res_ds = dataset
            max_res_window_size = rasterio.windows.shape(ds_window)
            min_res_window_size = rasterio.windows.shape(ds_window)
            max_extent_window_bounds = dataset.window_bounds(ds_window)
        elif rasterio.windows.shape(ds_window) > max_res_window_size:
            max_res_ds = dataset
            max_res_window_size = rasterio.windows.shape(ds_window)
        elif rasterio.windows.shape(ds_window) < min_res_window_size:
            max_extent_window_bounds = dataset.window_bounds(ds_window)
            min_res_window_size = rasterio.windows.shape(ds_window)

    # new bounds fitted to the highest resolution file
    out_bounds = max_res_ds.window_bounds(
        round_window(max_res_ds.window(*max_extent_window_bounds))
    )
    out_shape = rasterio.windows.shape(round_window(max_res_ds.window(*out_bounds)))

    return out_bounds, out_shape


def arr_stats(arr, weights=None, output=("min", "max", "sum", "mean", "count", "mode")):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Extracts statistics from input array (arr).
    It uses weights array as weights if provided.
    List of statistics to extract is defined in the output parameter as a list or a comma separated string.

    :param arr: Input array
    :param weights: Array providing weights for each pixel value. Used to calculate stats.
    :param output: List of values to extract, can be a list or comma separated string.
        Possible values are:
            - min
            - max
            - sum - sum of all un masked arr pixels, no weight applied
            - mean - average value, if weights are provided calculates a weighted average
            - std - standard deviation, if weights are provided uses weighted calculation
            - median - median values, WARNING doesn't use weights
            - count - number of pixels used in a group, can be different to total number if a mask is applied
            - weight_sum - sum of weights
            - mode - majority value, used for category variable.
    :return: dict with calculated stats values
    """
    # prepare output and make sure it is a list
    _output = output if type(output) in [list, tuple] else output.split()
    _known_outputs = (
        "min",
        "max",
        "sum",
        "mean",
        "std",
        "median",
        "count",
        "mode",
        "weight_sum",
    )  # todo handle more extractions
    if not any(elem in _output for elem in _known_outputs):
        raise Exception(
            "Output not defined properly, should define at least one known output %s"
            % str(_known_outputs)
        )

    out_vals = dict()
    # make sure array is a masked array
    _arr = np.ma.array(arr)
    if weights is not None:
        _weights = np.ma.masked_array(weights, mask=np.ma.getmask(arr))

    if any(elem in _output for elem in ("std", "min", "max", "sum", "median", "mode")):
        arr_compressed = _arr.compressed()
        if weights is not None:
            weights_compressed = _weights.compressed()

    if "mean" in _output:
        if weights is not None:
            ind = np.isnan(_arr) | np.isnan(weights) | (_arr <= -9999)
            out_vals["mean"] = np.ma.average(_arr[~ind], weights=weights[~ind])
        else:
            ind = np.isnan(_arr)
            out_vals["mean"] = np.ma.mean(_arr[~ind])

    if "std" in _output:
        arr_size = np.size(arr_compressed)
        if weights is not None:
            # combine mask from the arr and create a compressed arr
            weights_compressed = np.ma.array(weights, mask=_arr.mask).compressed()
            ind = (
                np.isnan(arr_compressed)
                | np.isnan(weights_compressed)
                | (arr_compressed <= -9999)
            )

            if arr_size == 1 or np.sum(weights_compressed[~ind] > 0) == 1:
                out_vals["std"] = np.int8(0)
            elif arr_size > 0 and np.sum(weights_compressed[~ind]) > 0:
                out_vals["std"] = np.sqrt(
                    np.cov(
                        arr_compressed[~ind], aweights=weights_compressed[~ind], ddof=0
                    )
                )
            else:
                out_vals["std"] = None
        else:
            if arr_size == 1:
                out_vals["std"] = np.int8(0)
            elif arr_size > 0:
                out_vals["std"] = np.sqrt(np.cov(arr_compressed, ddof=0))
            else:
                out_vals["std"] = None

    if "min" in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals["min"] = arr_compressed[~ind].min()

    if "max" in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals["max"] = arr_compressed[~ind].max()

    if "sum" in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals["sum"] = arr_compressed[~ind].sum()

    if "median" in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        out_vals["median"] = np.ma.median(arr_compressed[~ind])

    if "mode" in _output:
        ind = np.isnan(arr_compressed) | (arr_compressed <= -9999)
        if weights is not None:
            out_vals["mode"] = np.argmax(
                np.bincount(arr_compressed[~ind], weights=weights_compressed[~ind])
            )
        else:
            out_vals["mode"] = np.argmax(np.bincount(arr_compressed[~ind]))

    if "count" in _output:
        out_vals["count"] = int((~_arr.mask).sum())

    if "weight_sum" in _output and weights is not None:
        ind = np.isnan(weights) | (_arr <= -9999)
        weights_compressed = np.ma.array(
            weights[~ind], mask=_arr.mask[~ind]
        ).compressed()
        out_vals["weight_sum"] = weights_compressed.sum()

    # convert to regular py types from np types which can cause problems down the line like JSON serialisation
    out_vals = {k: v.item() for k, v in out_vals.items()}

    return out_vals


def arr_classes_count(arr, cls_def, weights=None, border_include="min"):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Counts the number of array values in a class (bin) defined by min and max value.

    :param arr: Input array
    :param cls_def: list(dict) - List of dictionaries with Class definitions. A class is defined by its min and max value.
        [{'min': val1, 'max': val2}, ...]
    :param weights: Array with weights to apply when counting pixels. If not defined pixels are counted as 1
    :param border_include: str [min|max|both|None] - Parameter defining how to handle the border values.
        Options min|max|both meaning to use "min", "max" or "both" values as a part of the class.
    :return: list(dict) - input Classes definition expanded with the pixel count for that class.
    """
    _weights = weights if weights is not None else 1
    cls_out = []

    for cls in cls_def:
        if border_include is None:
            cls["val_count"] = np.sum(
                np.logical_and(arr > cls["min"], arr < cls["max"]) * _weights
            )
        elif border_include.lower() == "min":
            cls["val_count"] = np.sum(
                np.logical_and(arr >= cls["min"], arr < cls["max"]) * _weights
            )
        elif border_include.lower() == "max":
            cls["val_count"] = np.sum(
                np.logical_and(arr > cls["min"], arr <= cls["max"]) * _weights
            )
        elif border_include.lower() == "both":
            cls["val_count"] = np.sum(
                np.logical_and(arr >= cls["min"], arr <= cls["max"]) * _weights
            )
        else:
            raise ValueError(
                'Parameter "border_include" not defined properly. Allowed values are "min", "max", "both" or None'
            )
        cls_out.append(cls)

    return cls_out


def arr_unpack(arr, scale=1, offset=0, nodata=None):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Converts the values in the array to native format by applying scale,
    offset and nodata (val or func).

    :param arr: array to be converted
    :param scale: conversion to native format scale factor
    :param offset: conversion to native format offset factor
    :param nodata: no data value or function to mask the dataset
    :return: np.array or np.ma.array converted (unpacked) to native values
    """
    # adjust for nodata
    if nodata:
        arr = apply_nodata(arr, nodata)
    # convert to native values
    if scale and scale != 1:
        arr = arr * scale
    if offset and offset != 0:
        arr = arr + offset

    return arr


def apply_nodata(arr, nodata):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Masks the array with nodata definition.
    If arr is masked array the two masks are combined.

    :param arr: array to be masked
    :param nodata: no data value, function or array to mask the dataset
    :return: np.ma.array masked with nodata
    """
    if isinstance(nodata, np.ndarray):
        return np.ma.array(arr, mask=nodata)
    elif callable(nodata):
        return np.ma.array(arr, mask=nodata(arr))
    elif type(nodata) in [list, tuple]:
        return np.ma.array(arr, mask=np.isin(arr, nodata))
    else:
        return np.ma.array(arr, mask=arr == nodata)


def geom_extract(
    geometry,
    indicator,
    indicator_name,
    stats_out=("mean", "std", "min", "max", "sum", "counts", "mode"),
    afi=None,
    afi_thresh=None,
    thresh_type=None,
    classification=None,
    indexes=None,
):
    """
    @author: Joint Research Centre - D5 Food Security - ASAP
    Extracts the indicator statistics on input geometry using the AFI as weights.

    Global variable SUPPRESS_ERRORS controls if a custom error (UnableToExtractStats)
    should be raised when it's not possible to extract stats with given parameters.
    By default it is set to suppress errors and only report a warning. This setup is
    for the use case when the function is called directly and can handle an empty output.
    The opposite case, when the errors are raised is used when this function is called in
    a multiprocessing pool and it's necessary to link a proper error message with
    a geometry/unit identifier.

    Handles heterogeneous datasets by using the tbx_util.raster.get_common_bounds_and_shape function.

    :param geometry: GeoJSON-like feature (implements __geo_interface__) – feature collection, or geometry.
    :param indicator: path to raster file or an already opened dataset (rasterio.DatasetReader)
        on which statistics are extracted
    :param indicator_name: name of indicator
    :param stats_out: definition of statistics to extract, the list is directly forwarded to function
        asap_toolbox.util.raster.arr_stats.
        Additionally, accepts "counts" keyword that calculates following values:
            - total - overall unit grid coverage
            - valid_data - indicator without nodata
            - valid_data_after_masking - indicator used for calculation
            - weight_sum - total mask sum
            - weight_sum_used - mask sum after masking of dataset nodata is applied
    :param afi: path to Area Fraction index or weights - path to raster file or
        an already opened dataset (rasterio.DatasetReader)
    :param afi_thresh: threshold to mask out the afi data
    :param tresh_type: type of afi_thresh ("Fixed" aka value or "percentile")
    :param classification: If defined, calculates the pixel/weight sums of each class defined.
        Defined as JSON dictionary with borders as list of min, max value pairs and
        border behaviour definition:
            {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: [min|max|both|None]
            }
    :return: dict with extracted stats divided in 3 groups:
        - stats - dict with calculated stats values (mean, std, min, max)
        - counts - dict with calculated count values (total; valid_data; valid_data_after_masking;
                    weight_sum; weight_sum_used)
        - classification - dict with border definitions and values
        {
            stats: {mean: val, std: min: val, max: val, ...}
            counts: {total: val, valid_data: valid_data_after_masking: val, weight_sum: val, ...}
            classification: {
                borders: ((min1, max1), (min2, max2), ..., (min_n, max_n)),
                border_include: val,
                values: (val1, val2, val3,...)
            }
        }
        raises UnableToExtractStats error if geom outside raster, if the geometry didn't catch any pixels
    """
    output = dict()
    # make sure inputs are opened
    indicator_ds = (
        indicator
        if isinstance(indicator, rasterio.DatasetReader)
        else rasterio.open(indicator)
    )
    rasters_list = [indicator_ds]
    if afi:
        afi_ds = afi if isinstance(afi, rasterio.DatasetReader) else rasterio.open(afi)
        rasters_list.append(afi_ds)

    # get unified read window, bounds and resolution if heterogeneous resolutions
    try:
        read_bounds, read_shape = get_common_bounds_and_shape([geometry], rasters_list)
    except rasterio.errors.WindowError:
        e_msg = "Geometry has no intersection with the indicator"
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return
        else:
            raise UnableToExtractStats(e_msg)

    # fetch indicator array
    indicator_arr = read_masked(
        ds=indicator_ds,
        mask=[geometry],
        window=indicator_ds.window(*read_bounds),
        indexes=indexes,
        use_pixels="CENTER",
        out_shape=read_shape,
    )
    if indicator_arr.ndim == 3 and indicator_arr.shape[0] == 1:
        indicator_arr = indicator_arr[0]
    nodatavals = indicator_ds.nodatavals

    # fpar values must be between 0 and 100
    # See https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_fpar_500m/README.md
    if indicator_name == "fpar":
        indicator_arr[(indicator_arr < 0) | (indicator_arr > 100)] = 0
    # convert Kelvin to Celsius
    # see https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_AgERA5/README.md
    elif indicator_name in ["tmin", "tmax", "tavg"]:
        indicator_arr = indicator_arr - 273.15
    elif indicator_name == "ndvi":
        # data cleaning
        indicator_arr.mask |= (indicator_arr > 250) | (indicator_arr < 50)
    if indicator_name in ["sos", "eos"]:
        max_value = 365
        if np.ptp(indicator_arr.compressed()) > max_value / 2:
            # Adjust values for wrap-around
            adjusted_data = np.where(
                indicator_arr < max_value / 2, indicator_arr + max_value, indicator_arr
            )
            # Convert back to MaskedArray and reapply the original mask
            indicator_arr = np.ma.masked_array(adjusted_data, mask=indicator_arr.mask)

    geom_mask = indicator_arr.mask
    # skip extraction if no pixels caught by geom
    if np.all(geom_mask):
        e_msg = "No pixels caught by geometry"
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return
        else:
            raise UnableToExtractStats(e_msg)
    # convert pixel values if ENVI file
    if nodatavals:
        _dtype_conversion = dict(nodata=nodatavals)
    if _dtype_conversion:
        indicator_arr = arr_unpack(indicator_arr, **_dtype_conversion)
    valid_data_mask = indicator_arr.mask

    # fetch mask array
    if afi:
        afi_arr = read_masked(
            ds=afi_ds,
            mask=[geometry],
            indexes=None,
            window=afi_ds.window(*read_bounds),
            use_pixels="CENTER",
            out_shape=read_shape,
        )
        if afi_arr.ndim == 3 and afi_arr.shape[0] == 1:
            afi_arr = afi_arr[0]

        if afi_thresh is not None:
            # afi must be between 0 and 100
            # https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_crop_AFIs_ESA_WC/README.md
            afi_arr[(afi_arr < 0) | (afi_arr > 100)] = 0
            if thresh_type == "Fixed":
                afi_arr[
                    ~np.isnan(afi_arr) & (afi_arr <= afi_thresh) & ~afi_arr.mask
                ] = 0

            elif thresh_type == "Percentile":
                m_afi_arr = afi_arr[~np.isnan(afi_arr) & (afi_arr > 0) & ~afi_arr.mask]

                if len(m_afi_arr) > 0:
                    thresh_PT = np.percentile(m_afi_arr, afi_thresh)

                    afi_arr[
                        ~np.isnan(afi_arr) & (afi_arr <= thresh_PT) & ~afi_arr.mask
                    ] = 0

            afi_arr = np.ma.array(afi_arr, mask=(afi_arr.mask + (afi_arr == 0)))

        # convert pixel values if ENVI file
        if afi_ds.nodatavals:
            _dtype_conversion = dict(nodata=afi_ds.nodatavals)
        if _dtype_conversion:
            afi_arr = arr_unpack(afi_arr, **_dtype_conversion)
        # apply the afi mask nodata mask to the dataset
        indicator_arr = np.ma.array(indicator_arr, mask=(afi_arr.mask + (afi_arr == 0)))

    # check if any data left after applying all the masks
    if np.sum(~indicator_arr.mask) == 0:
        e_msg = "No data left after applying all the masks, mask sum == 0"
        if SUPPRESS_ERRORS:
            # log.warning('Skipping extraction! ' + e_msg)
            return output
        else:
            raise UnableToExtractStats(e_msg)

    # extractions
    if any(val in ("min", "max", "mean", "sum" "std", "mode") for val in stats_out):
        output["stats"] = arr_stats(indicator_arr, afi_arr if afi else None, stats_out)
        # Apply modulo max_value to all fields in stats
        if indicator_name in ["sos", "eos"]:
            output["stats"] = {
                key: (value % max_value) if isinstance(value, (int, float)) else value
                for key, value in output["stats"].items()
            }
        if indicator_name == "ndvi":
            # rescale ndvi
            # https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/global_MOD09CMG/README.md
            output["stats"] = {
                key: ((value - 50) / 200) for key, value in output["stats"].items()
            }

    if "counts" in stats_out:
        output["counts"] = dict()
        # total - overall unit grid coverage
        output["counts"]["total"] = int((~geom_mask).sum())
        # valid_data - indicator without nodata
        output["counts"]["valid_data"] = int(np.sum(~valid_data_mask))
        if afi:
            output["counts"]["valid_data_after_masking"] = int(
                np.sum(~indicator_arr.mask)
            )
            # weight_sum - total mask sum
            output["counts"]["weight_sum"] = afi_arr.sum()
            if type(output["counts"]["weight_sum"]) == np.uint64:
                output["counts"]["weight_sum"] = int(output["counts"]["weight_sum"])
            # weight_sum_used - mask sum after masking of dataset nodata is applied
            afi_arr_compressed = np.ma.array(
                afi_arr, mask=indicator_arr.mask
            ).compressed()
            output["counts"]["weight_sum_used"] = afi_arr_compressed.sum()
            if type(output["counts"]["weight_sum_used"]) == np.uint64:
                output["counts"]["weight_sum_used"] = int(
                    output["counts"]["weight_sum_used"]
                )

    if classification:
        cls_def = [
            {"min": _min, "max": _max} for _min, _max in classification["borders"]
        ]
        classification_out = classification.copy()
        classification_out["border_include"] = classification.get(
            "border_include", "min"
        )
        class_res = arr_classes_count(
            indicator_arr,
            cls_def=cls_def,
            weights=afi_arr if afi else None,
            border_include=classification_out["border_include"],
        )
        classification_out["values"] = [i["val_count"] for i in class_res]
        output["classification"] = classification_out

    return output


def _crop_mask_path(crop: str) -> str:
    if crop == "maize":
        crop_mask_file = "crop_mask_maize_WC.tif"
    elif crop == "wheat":
        crop_mask_file = "crop_mask_winter_spring_cereals_WC.tif"
    else:
        crop_mask_file = "crop_mask_generic_asap.tif"

    for base in (
        os.path.join(AGML_ROOT, "crop_masks"),
        os.path.join(REPO_DIR, "data_preparation", "global_crop_AFIs_ESA_WC"),
    ):
        path = os.path.join(base, crop_mask_file)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Crop mask not found: {crop_mask_file}")


def _netcdf_dataset_path(indicator_file: str, indicator_name: str) -> str:
    if indicator_name in AGERA5_VARIABLES:
        variable = AGERA5_VARIABLES[indicator_name]
    elif indicator_name in LPJML_VARIABLES:
        variable = LPJML_VARIABLES[indicator_name]
    else:
        raise KeyError(f"No NetCDF variable mapping for indicator {indicator_name!r}")
    return "netcdf:{indicator_file}:{variable}".format(
        indicator_file=indicator_file,
        variable=variable,
    )


def process_file(
    indicator_file,
    crop,
    indicator_name,
    geometries,
    is_time_series,
    is_categorical,
    band_index=None,
    date_str=None,
):
    """
    @author: Guanyuan Shuai
    Process one indicator raster file. Handles .nc or .tif files.

    :param file: path to indicator raster file.
    :param crop: crop name
    :param indicator_name: indicator name
    :param geometry: geometry
    :param is_time_series: flag to indicate whether data is static or time series
    :param is_categorical: flag to indicator whether data is categorical or continuous
    :return a dataframe with data from given raster file aggregated to admin units
    """
    crop_mask_path = _crop_mask_path(crop)

    basename = os.path.basename(indicator_file)
    fname, ext = os.path.splitext(basename)
    if ext == ".nc":
        indicator_file = _netcdf_dataset_path(indicator_file, indicator_name)

    if is_categorical:
        aggr = "mode"
    else:
        aggr = "mean"

    if is_time_series:
        if date_str is None:
            date_str = fname[-8:]
        col_names = ["crop_name", "adm_id", "date", indicator_name]
    else:
        date_str = None
        col_names = ["crop_name", "adm_id", indicator_name]

    ############################################
    # get predictor value for each admin region
    ############################################
    df = pd.DataFrame(columns=col_names)
    for adm_id, geometry in geometries.items():
        stats = geom_extract(
            geometry,
            indicator_file,
            indicator_name,
            stats_out=(aggr,),
            afi=crop_mask_path,
            afi_thresh=0,
            thresh_type="Fixed",
            indexes=band_index,
        )
        if (stats is not None) and ("stats" in stats) and (aggr in stats["stats"]):
            aggr_val = stats["stats"][aggr]
            if is_time_series:
                data_row = [crop, adm_id, date_str, aggr_val]
            else:
                data_row = [crop, adm_id, aggr_val]
            df.loc[len(df.index)] = data_row

    return df


def get_time_series_files(data_path, year=2000):
    """
    @author: Dilli R. Paudel
    Returns a list of raster files for the given year.

    :param data_path: path to directory containing raster files
    :param year: year of interest
    :return: a list of rasters for given year
    """
    files = []
    for f in os.listdir(data_path):
        fname, ext = os.path.splitext(f)
        if not (ext == ".tif" or ext == ".nc"):
            continue

        # we expect the last part of filename to be YYYYMMDD
        date_str = fname[-8:]
        if int(date_str[:4]) == year:
            files.append(f)

    return files


def iter_lpjml_time_slices(indicator_dir, indicator_name, year):
    """Yield (file_path, band_index, date_str) for one LPJmL harvest year."""
    if os.path.isdir(indicator_dir):
        for fname in get_time_series_files(indicator_dir, year=year):
            date = os.path.splitext(fname)[0][-8:]
            yield os.path.join(indicator_dir, fname), None, date

    for nc_path in (
        os.path.join(indicator_dir, f"{indicator_name}.nc"),
        os.path.join(os.path.dirname(indicator_dir), f"{indicator_name}.nc"),
    ):
        if not os.path.isfile(nc_path):
            continue
        band_index = year - 2000 + 1
        yield nc_path, band_index, f"{year}0101"
        return


def iter_time_series_inputs(indicator_dir, indicator_name, year, pred_source):
    if pred_source == "LPJmL":
        return list(iter_lpjml_time_slices(indicator_dir, indicator_name, year))
    if not os.path.isdir(indicator_dir):
        return []
    return [
        (os.path.join(indicator_dir, fname), None, os.path.splitext(fname)[0][-8:])
        for fname in get_time_series_files(indicator_dir, year=year)
    ]


def get_admin_geometries(region):
    shp_path = os.path.join(PATH_POLYGONS_DIR, region, f"{region}.shp")
    if os.path.isfile(shp_path):
        geo_df = get_shapes_from_polygons(region=region)
    else:
        geo_df = get_shapes(region=region)
    geo_df = geo_df[["adm_id", "geometry"]]
    return {
        adm_id: geo_df[geo_df["adm_id"] == adm_id]["geometry"].values[0]
        for adm_id in geo_df["adm_id"].unique()
    }


def get_shapes(region="US"):
    """
    @author: Dilli R. Paudel
    Get admin unit boundaries.

    :param region: region code or 2-letter country code
    :return: a dataframe with adm_id and boundaries
    """
    sel_shapes = pd.DataFrame()
    if region == "EU":
        geo_df = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_EU.zip")
        )
        for cn in EU_COUNTRIES:
            cn_shapes = geo_df[
                (geo_df[EU_COUNTRY_CODE_KEY] == cn)
                & (geo_df[EU_ADMIN_LEVEL_KEY] == EU_COUNTRIES[cn])
            ]
            sel_shapes = pd.concat([sel_shapes, cn_shapes], axis=0)

        sel_shapes["adm_id"] = sel_shapes["NUTS_ID"]
    elif region in EU_COUNTRIES:
        geo_df = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_EU.zip")
        )
        sel_shapes = geo_df[
            (geo_df[EU_COUNTRY_CODE_KEY] == region)
            & (geo_df[EU_ADMIN_LEVEL_KEY] == EU_COUNTRIES[region])
        ]
        sel_shapes["adm_id"] = sel_shapes["NUTS_ID"]

    elif region == "AR":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_AR.zip")
        )
        sel_shapes["adm_id"] = sel_shapes["ADM2_PCODE"]
    elif region == "AU":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_AU.zip")
        )
        sel_shapes["adm_id"] = "AU" + "-" + sel_shapes["AAGIS"].astype(str)
    elif region == "BR":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_BR.zip")
        )
        sel_shapes["adm_id"] = sel_shapes["ADM2_PCODE"]
    elif region == "CN":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_CN.zip")
        )
        sel_shapes["adm_id"] = sel_shapes["ADM1_PCODE"]
    # FEWSNET countries: Already have adm_id
    elif region == "FEWSNET":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_FEWSNET.zip")
        )
    elif region in FEWSNET_COUNTRIES:
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_FEWSNET.zip")
        )
        sel_shapes = sel_shapes[sel_shapes["adm_id"].str[:2] == region]
    # IN: Already has adm_id
    elif region == "IN":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_IN.zip")
        )
    # ML: Already has adm_id
    elif region == "ML":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_ML.zip")
        )
    # MX: Already has adm_id
    elif region == "MX":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_MX.zip")
        )
        # adm_id in shapefile has a hyphen. Yield data does not have one.
        sel_shapes["adm_id"] = sel_shapes["adm_id"].str.replace("-", "")
    elif region == "US":
        sel_shapes = gpd.read_file(
            os.path.join(AGML_ROOT, "shapefiles", "shapefiles_US.zip")
        )
        sel_shapes["adm_id"] = (
            "US" + "-" + sel_shapes["STATEFP"] + "-" + sel_shapes["COUNTYFP"]
        )

    # Project to EPSG 4326
    # shapes for BR don't have crs info. See #343.
    if region not in ["BR"]:
        sel_shapes = sel_shapes.to_crs(4326)

    return sel_shapes


def process_indicators(crop, region, sel_indicators, data_dir=None, output_dir=None):
    """
    @author: Guanyuan Shuai
    Process predictors or indicators.

    :param crop: crop name
    :param region: region code or 2-letter country code
    :param sel_indicators: a list of indicators to process
    """
    data_dir = DATA_DIR if data_dir is None else data_dir
    output_dir = OUTPUT_DIR if output_dir is None else output_dir
    geometries = get_admin_geometries(region)

    #################Loop over each crop, year, and variable##########################
    ##########setup crop mask file###################
    for indicator in sel_indicators:
        pred_source = ALL_INDICATORS[indicator]["source"]
        is_time_series = ALL_INDICATORS[indicator]["is_time_series"]
        is_categorical = ALL_INDICATORS[indicator]["is_categorical"]
        output_path = os.path.join(output_dir, crop, region, indicator)
        os.makedirs(output_path, exist_ok=True)

        # Time series data
        if is_time_series:
            indicator_dir = os.path.join(data_dir, pred_source, indicator)
            result_final = pd.DataFrame()
            for yr in range(START_YEAR, END_YEAR + 1):
                print("Start working on", crop, region, indicator, yr)
                slices = iter_time_series_inputs(
                    indicator_dir, indicator, yr, pred_source
                )
                print("There are " + str(len(slices)) + " files!")
                if len(slices) == 0:
                    continue

                start_time = time.time()
                slices = sorted(slices, key=lambda item: item[2])
                with mp.Pool(processes=None, initializer=init_worker_logging) as pool:
                    # NOTE: multiprocessing using a target function with multiple arguments.
                    # Based on the answer to
                    # https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
                    dfs = pool.starmap(
                        process_file,
                        zip(
                            [item[0] for item in slices],
                            repeat(crop),
                            repeat(indicator),
                            repeat(geometries),
                            repeat(is_time_series),
                            repeat(is_categorical),
                            [item[1] for item in slices],
                            [item[2] for item in slices],
                        ),
                    )
                    result_yr = pd.concat(dfs, axis=0)
                    result_final = pd.concat([result_final, result_yr], axis=0)
            result_final = result_final.round(3)
            out_csv = "_".join([indicator, crop, region]) + ".csv"
            result_final.to_csv(os.path.join(output_path, out_csv), index=False)

            m, s = divmod((time.time() - start_time), 60)
            h, m = divmod(m, 60)

            print("Time used: %02d:%02d:%02d" % (h, m, s))

        # Static data
        else:
            indicator_dir = os.path.join(data_dir, pred_source)
            print("Start working on", crop, region, indicator)
            files = os.listdir(indicator_dir)

            # for crop calendar, need to filter by crop as well
            if indicator in ["sos", "eos"]:
                files = [f for f in files if (crop in f) and (indicator in f)]
            else:
                files = [f for f in files if (indicator in f)]

            # should be one raster file
            assert len(files) == 1

            start_time = time.time()
            df = process_file(
                os.path.join(indicator_dir, files[0]),
                crop,
                indicator,
                geometries,
                is_time_series,
                is_categorical,
            )

            out_csv = "_".join([indicator, crop, region]) + ".csv"
            df.to_csv(os.path.join(output_path, out_csv), index=False)

            m, s = divmod((time.time() - start_time), 60)
            h, m = divmod(m, 60)

            print("Done for", crop, region, indicator)
            print("Time used: %02d:%02d:%02d" % (h, m, s))


def discover_regions(crop: str) -> list[str]:
    """List country/region codes to process when ``-r`` is not passed.

    Prefer subdirectories of ``cybench/data/{crop}/``; fall back to ``DATASETS``.
    """
    crop_dir = os.path.join(PATH_DATA_DIR, crop)
    if os.path.isdir(crop_dir):
        regions = sorted(
            cc
            for cc in os.listdir(crop_dir)
            if os.path.isdir(os.path.join(crop_dir, cc))
        )
        if regions:
            return regions
    return list(DATASETS.get(crop, []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="predictor_data_prep.py", description="Prepare CY-Bench predictor data"
    )
    parser.add_argument("-c", "--crop")
    parser.add_argument("-r", "--region", type=str, nargs="+", help="List of regions")
    parser.add_argument("-i", "--indicator", nargs="+")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Root directory with predictor rasters (default: AGML predictors path)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory for region-aggregated CSV output",
    )
    args = parser.parse_args()

    data_dir = args.data_dir if args.data_dir is not None else DATA_DIR
    output_dir = args.output_dir if args.output_dir is not None else OUTPUT_DIR
    if args.crop is not None:
        sel_crops = [args.crop]
    else:
        sel_crops = CROPS

    sel_regions = None
    if args.region is not None:
        sel_regions = args.region

    if args.indicator is not None:
        sel_indicators = args.indicator
    else:
        sel_indicators = list(ALL_INDICATORS.keys())

    for crop in sel_crops:
        regions = sel_regions if sel_regions is not None else discover_regions(crop)
        print(f"Processing {crop} for {len(regions)} region(s): {', '.join(regions)}")

        for cn in regions:
            print("Working on", crop, cn)
            process_indicators(
                crop, cn, sel_indicators, data_dir=data_dir, output_dir=output_dir
            )
