"""
feature_design.py — define custom features for the CyBench pipeline.

HOW TO ADD A NEW FEATURE
-------------------------
1. Write a function below (see the examples for the expected signature).
2. Add it to the FEATURE_FUNCTIONS dict at the bottom of this file.
3. Reference its key in your dataset config under `create:`.

Function signature
------------------
Every function must accept exactly these arguments:

    def my_feature(df, col, group_keys, crop_params):
        ...
        return <pd.Series>

    df          — the time series DataFrame for the current source
    col         — name of the input column (set via `input:` in the config)
    group_keys  — [KEY_LOC, KEY_YEAR], use this when you need a groupby
    crop_params — crop config values, e.g. crop_params.gdd.tbase

The function must return a pd.Series with the same index as df.
"""

import pandas as pd


# --------------------------------------------------------------------------- #
# Feature functions                                                            #
# --------------------------------------------------------------------------- #

def cumulative_sum(df, col, group_keys, crop_params):
    """Running total of `col`, reset at the start of each (loc, year) season."""
    return df.groupby(group_keys, observed=True)[col].cumsum()


def growing_degree_days(df, col, group_keys, crop_params):
    """Daily growing degree days: clip(tavg - tbase, 0, upper_limit).

    Reads tbase and upper_limit from the crop config:
        gdd:
          tbase: 10.0
          upper_limit: 30.0
    """
    return (df[col] - crop_params.gdd.tbase).clip(0.0, crop_params.gdd.upper_limit)


# Add your own function here, then add it to FEATURE_FUNCTIONS below.
# def my_feature(df, col, group_keys, crop_params):
#     return ...


# --------------------------------------------------------------------------- #
# Register your functions here                                                 #
# key   = the `type:` value you use in the dataset config                     #
# value = the function to call                                                 #
# --------------------------------------------------------------------------- #

FEATURE_FUNCTIONS = {
    "cumsum": cumulative_sum,
    "gdd":    growing_degree_days,
    # "my_feature": my_feature,
}