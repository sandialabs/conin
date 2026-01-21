from math import nan
from dataclasses import dataclass


@dataclass
class PGMSolution:

    solution: dict = None
    value: float = nan
