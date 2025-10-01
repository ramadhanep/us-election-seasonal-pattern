import pandas as pd
from functools import lru_cache
from typing import Optional, Dict


@lru_cache(maxsize=1)
def load_credentials(path: str = "password.csv") -> Dict[str, str]:
    df = pd.read_csv(path, dtype={"pin": str, "name": str})
    df = df.dropna(subset=["pin", "name"])  # basic hygiene
    # Normalize: strip spaces
    df["pin"] = df["pin"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    return dict(zip(df["pin"], df["name"]))


def verify_pin(pin: str, path: str = "password.csv") -> Optional[str]:
    pin = (pin or "").strip()
    if not pin:
        return None
    creds = load_credentials(path)
    return creds.get(pin)

