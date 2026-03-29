import os

# Project root dir
PROJECT_DIR = os.path.abspath(os.path.join(__file__, os.pardir))

# Repository root dir
REPO_DIR = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))

# Path to folder where data is stored
PATH_DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(PATH_DATA_DIR, exist_ok=True)

# PATH to the config folder
CONF_DIR = os.path.join(PROJECT_DIR, "conf")

# Path to folder where output is stored
PATH_OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
os.makedirs(PATH_OUTPUT_DIR, exist_ok=True)

# Path to folder where benchmark results
PATH_RESULTS_DIR = os.path.join(PATH_OUTPUT_DIR, "runs")
os.makedirs(PATH_RESULTS_DIR, exist_ok=True)

PATH_POLYGONS_DIR = os.path.join(REPO_DIR, "cybench", "data", "polygons")

DATASETS = {
    "maize": [
        "AO", "AR", "AT", "BE", "BF", "BG", "BR", "CN", "CZ", "DE",
        "DK", "EL", "ES", "ET", "FR", "HR", "HU",
        "IN", "IT", "LS", "LT", "MG", "ML", "MW", "MX", "MZ",
        "NE", "NL", "PL", "PT", "RO", "SE", "SK", "SN", "TD", "US",
        "ZA", "ZM",
    ],
    "wheat": [
        "AR", "AT", "AU", "BE", "BG", "BR", "CN", "CZ", "DE", "DK",
        "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IN", "IT",
        "LT", "LV", "NL", "PL", "PT", "RO", "SE", "SK", "US",
    ],
}

CONTINENT_DICT = {
    "Africa": ["AO", "BF", "ET", "LS", "MG", "ML", "MW", "MZ", "NE", "SN", "TD", "ZA", "ZM"],
    "Asia": ["CN", "IN"],
    "Europe": ["AT", "BE", "BG", "CZ", "DE", "DK", "EL", "ES", "FR", "HR", "HU", "IT", "LT", "NL", "PL", "PT", "RO", "SE", "SK"],
    "North America": ["MX", "US"],
    "South America": ["AR", "BR"]
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
# Crop calendar entries: start of season, end of season.
# doy = day of year (1 to 366).
CROP_CALENDAR_DOYS = ["sos", "eos"]
CROP_CALENDAR_DATES = ["sos_date", "eos_date", "start_of_sequence_date", "end_of_sequence_date"]
