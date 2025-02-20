import numpy as np
from typing import Union


def calculate_distance(r1: Union[np.ndarray, float], r2: Union[np.ndarray, float], 
                      z1: Union[np.ndarray, float], z2: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """
    Compute the Euclidean distance between two points (r1, z1) and (r2, z2).
    Works with both scalar values and numpy arrays.

    :param r1: Radius coordinate(s) of the first point(s)
    :param r2: Radius coordinate(s) of the second point(s)
    :param z1: Z coordinate(s) of the first point(s)
    :param z2: Z coordinate(s) of the second point(s)
    :return: Euclidean distance(s)
    """
    return np.sqrt((r2 - r1) ** 2 + (z2 - z1) ** 2)
