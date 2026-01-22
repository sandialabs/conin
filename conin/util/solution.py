from math import nan
from dataclasses import dataclass


@dataclass
class MPESolution:

    states: dict = None
    log_value: float = nan
