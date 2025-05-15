import os
from typing import Union

import numpy as np
from .figures import Figure, Direction

ImageNP = np.ndarray
CellsImages = list[list[ImageNP]]
FigureBoard = list[list[Figure]]
DirectionBoard = list[list[Direction]]
Corners = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | np.ndarray[int]
Inventory = dict[Figure, int]
Box = tuple[int, int, int, int]
FilePath = Union[str, os.PathLike[str]]