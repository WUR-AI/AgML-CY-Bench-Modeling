import os

from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

from omegaconf import DictConfig

# Project root dir
CONFIG_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Repository root dir
REPO_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(CONFIG_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# Path to folder where output is stored
PATH_OUTPUT_DIR = os.path.join(CONFIG_DIR, "output")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

# Path to folder where benchmark results
PATH_RESULTS_DIR = os.path.join(PATH_OUTPUT_DIR, "runs")
os.makedirs(PATH_RESULTS_DIR, exist_ok=True)

PATH_POLYGONS_DIR = os.path.join(REPO_DIR, "cybench", "data", "polygons")

DATASETS = {
    "maize": [
        "AO", "AR", "AT", "BE", "BF", "BG", "BR", "CN", "CZ", "DE",
        "DK", "EE", "EL", "ES", "ET", "FI", "FR", "HR", "HU", "IE",
        "IN", "IT", "LS", "LT", "LV", "MG", "ML", "MW", "MX", "MZ",
        "NE", "NL", "PL", "PT", "RO", "SE", "SK", "SN", "TD", "US",
        "ZA", "ZM",
    ],
    "wheat": [
        "AR", "AT", "AU", "BE", "BG", "BR", "CN", "CZ", "DE", "DK",
        "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IN", "IT",
        "LT", "LV", "NL", "PL", "PT", "RO", "SE", "SK", "US",
    ],
}

# key used for 2-letter country code
KEY_COUNTRY = "country_code"
# Key used for the location index
KEY_LOC = "adm_id"
# Key used for the year index
KEY_YEAR = "year"
# Key used for yield targets
KEY_TARGET = "yield"
# Key used for dates matching observations
KEY_DATES = "dates"
# Key used for crop season data
KEY_CROP_SEASON = "crop_season"
# Key used for combined input features
KEY_COMBINED_FEATURES = "combined_features"

# Minimum and maximum year in input data.
# Used to add years to crop calendar data.
MIN_INPUT_YEAR = 2000
MAX_INPUT_YEAR = 2023

CROP_YIELD_RANGES = {
    "wheat": {
        "min": 0,
        "max": 9,
    },
    "maize": {"min": 0, "max": 14},
}

# Soil properties
SOIL_PROPERTIES = ["awc", "bulk_density"]  # , "drainage_class"]

# Location properties
LOCATION_PROPERTIES = ["latitude", "longitude"]

# Static predictors. Add more when available
STATIC_PREDICTORS = SOIL_PROPERTIES

# Weather indicators
METEO_INDICATORS = ["tmin", "tmax", "tavg", "prec", "cwb", "rad"]

# Remote sensing indicators.
# Keep them separate because they have different temporal resolution
RS_FPAR = "fpar"
RS_NDVI = "ndvi"

# Soil moisture indicators: surface moisture, root zone moisture
SOIL_MOISTURE_INDICATORS = ["ssm"]  # , "rsm"]

TIME_SERIES_INPUTS = {
    "meteo": METEO_INDICATORS,
    "fpar": [RS_FPAR],
    "ndvi": [RS_NDVI],
    "soil_moisture": SOIL_MOISTURE_INDICATORS,
}

# Time series predictors
TIME_SERIES_PREDICTORS = sum(TIME_SERIES_INPUTS.values(), [])

# Aggregation functions
TIME_SERIES_AGGREGATIONS = {
    "tmin": "min",
    "tmax": "max",
    "tavg": "mean",
    "prec": "sum",
    "cwb": "sum",
    "rad": "mean",
    RS_FPAR: "mean",
    RS_NDVI: "mean",
    "ssm": "mean",
}

# All predictors. Add more when available
ALL_PREDICTORS = STATIC_PREDICTORS + TIME_SERIES_PREDICTORS

# Crop calendar entries: start of season, end of season.
# doy = day of year (1 to 366).
CROP_CALENDAR_DOYS = ["sos", "eos"]
CROP_CALENDAR_DATES = ["sos_date", "eos_date", "start_of_sequence_date", "end_of_sequence_date"]

# Feature design
# Base temperature for corn and wheat for growing degree days wheat:0 maize:10.
# From @poudelpratishtha.
GDD_BASE_TEMP = {
    "maize": 10,
    "wheat": 0,
}

GDD_UPPER_LIMIT = {
    "maize": 35,
    "wheat": None,
}


@dataclass
class DatasetConfig:
    """
    Dataset configuration class for type checking the yaml configuration under conf/dataset/...
    """
    crop: str
    country: str
    name: str
    min_year: int
    max_year: int
    target: Dict[str, Any]
    non_temporal: Dict[str, Any]
    temporal: Dict[str, Any]
    framework: str

@dataclass
class EvaluationConfig:
    """
    Evaluation configuration class for type checking the yaml configuration under conf/evaluation/...
    """
    name: str
    metrics: List[str]
    residual: bool = False

@dataclass
class ExperimentConfig:
    """
    Experiment configuration class for type checking the final conf/config.py
    """
    dataset: DatasetConfig
    model: Dict[str, Any] # TODO customize as well
    evaluation: EvaluationConfig
    validation: Dict[str, Any]
    hp_search: Dict[str, Any] # TODO customize as well
    experiment: Dict[str, Any] # TODO customize as well
    run: Dict[str, Any]
