"""Load raw BDG2 electricity meter data and building metadata into long format."""
import pandas as pd

RAW_ELECTRICITY = "data/raw/electricity_cleaned.csv"
RAW_METADATA = "data/raw/metadata.csv"


def load_metadata(path: str = RAW_METADATA) -> pd.DataFrame:
    meta = pd.read_csv(path)
    return meta.set_index("building_id")


def load_electricity_wide(path: str = RAW_ELECTRICITY) -> pd.DataFrame:
    """Timestamp-indexed, one column per building_id."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    return df


def select_buildings(wide: pd.DataFrame, max_buildings: int = 200, max_missing_frac: float = 0.05) -> list[str]:
    """Rank buildings by completeness, return the most complete `max_buildings` ids."""
    missing_frac = wide.isna().mean().sort_values()
    candidates = missing_frac[missing_frac <= max_missing_frac]
    return list(candidates.index[:max_buildings])


def to_long(wide: pd.DataFrame, building_ids: list[str]) -> pd.DataFrame:
    sub = wide[building_ids].copy()
    long = sub.reset_index().melt(id_vars="timestamp", var_name="building_id", value_name="electricity")
    return long.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
