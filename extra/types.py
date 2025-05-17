import os
from typing import Union

import numpy as np
from .figures import Figure, Direction

# Image stored as numpy array
ImageNP = np.ndarray

# 9x9 list with images of cells
CellsImages = list[list[ImageNP]]

# 9x9 list with Figure enums
FigureBoard = list[list[Figure]]

# 9x9 list with Direction enums
DirectionBoard = list[list[Direction]]

# Coordinates of 4 corners
Corners = tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | np.ndarray[int]

# Count of each figure
Inventory = dict[Figure, int]

# x, y, w, h
Box = tuple[int, int, int, int]

FilePath = Union[str, os.PathLike[str]]