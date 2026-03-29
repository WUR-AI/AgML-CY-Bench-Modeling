import numpy as np
import pandas as pd


def dummy_encode(
    df: pd.DataFrame,
    feature: str,
    classes: list,
    drop_original: bool = True,
    unknown_value: float = 0.0,
) -> pd.DataFrame:
    """
    One-hot encode a categorical feature using a fixed, global class list.

    Output columns are named: <feature>_<class_name>

    Parameters
    ----------
    df : pd.DataFrame
    feature : str
        Categorical column to encode.
    classes : list
        List of allowed class labels (strings or ints).
    drop_original : bool
        Whether to drop the original column.
    unknown_value : float
        Value assigned if a value is not in `classes`.
    """

    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found.")

    values = df[feature]

    for cls in classes:
        col_name = f"{feature}_{cls}"
        df[col_name] = (values == cls).astype(float)

    # handle unknown categories explicitly
    known_mask = values.isin(classes)
    if not known_mask.all():
        for cls in classes:
            df.loc[~known_mask, f"{feature}_{cls}"] = unknown_value

    if drop_original:
        df = df.drop(columns=[feature])

    return df


def cyclical_encode(
    df: pd.DataFrame,
    feature: str,
    period: float,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Encode a cyclic variable onto the unit circle.
    Example: day-of-year with period=365.
    """
    if feature not in df.columns:
        raise KeyError(f"Feature '{feature}' not found in DataFrame.")

    x = df[feature].to_numpy(dtype=float)

    angle = 2.0 * np.pi * x / period
    df[f"{feature}_sin"] = np.sin(angle)
    df[f"{feature}_cos"] = np.cos(angle)

    if drop_original:
        df = df.drop(columns=[feature])

    return df


def spherical_encode(
    df: pd.DataFrame,
    lat_feature: str,
    lon_feature: str,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Project latitude/longitude onto the unit sphere.
    Output: <lat>_<lon>_x, _y, _z
    """
    if lat_feature not in df.columns or lon_feature not in df.columns:
        raise KeyError("Latitude or longitude feature not found.")

    lat = np.deg2rad(df[lat_feature].to_numpy(dtype=float))
    lon = np.deg2rad(df[lon_feature].to_numpy(dtype=float))

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    prefix = "loc" #f"{lat_feature}_{lon_feature}"
    df[f"{prefix}_x"] = x
    df[f"{prefix}_y"] = y
    df[f"{prefix}_z"] = z

    if drop_original:
        df = df.drop(columns=[lat_feature, lon_feature])

    return df


def feature_transform(df_x, transform):
    transform_type = transform["type"]

    if transform_type == "dummy":
        df_x = dummy_encode(df=df_x,
                            feature=transform["feature"],
                            classes=transform["classes"],)

    elif transform_type == "cyclical":
        df_x = cyclical_encode(
            df_x,
            feature=transform["feature"],
            period=transform["period"],
        )

    elif transform_type == "spherical":
        df_x = spherical_encode(
            df_x,
            lat_feature=transform["lat"],
            lon_feature=transform["lon"],
        )

    else:
        raise ValueError(f"Unknown transform type '{transform_type}'")
    return df_x