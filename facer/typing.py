from abc import abstractmethod
from typing import Any, Callable, Iterable, Iterator

import numpy as np
from numpy.typing import NDArray


NumPyNumericArray = NDArray[np.integer[Any] | np.floating[Any]]

"""http://dlib.net/python/index.html"""


class Vector:
    """This object represents the mathematical idea of a column vector."""

    def __init__(self) -> None: ...

    def resize(self, arg0: int) -> None: ...
    def set_size(self, arg0: int) -> None: ...

    shape: int


class Point:
    """This object represents a single point of integer coordinates that maps directly to a dlib::point."""

    def __init__(self, x: int, y: int) -> None:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> int: ...

    @abstractmethod
    def normalize(self) -> Vector: ...

    x: int
    y: int


class Points:
    """An array of point objects."""

    @abstractmethod
    def __init__(self) -> None: ...

    def __init__(self, arg0: "Points") -> None: ...

    def __init__(self, arg0: Iterable["Points"]) -> None: ...

    def __init__(self, initial_size: int) -> None: ...

    @abstractmethod
    def __iter__(self) -> Iterator[Point]: ...

    def append(self, x: Point) -> None:
        """Add an item to the end of the list"""
        ...

    def clear(self) -> None: ...

    @abstractmethod
    def count(self, x: Point) -> int:
        """Return the number of times x appears in the list"""
        ...

    @abstractmethod
    def extend(self, L: "Points") -> None:
        """Extend the list by appending all the items in the given list"""
        ...

    @abstractmethod
    def extend(self, arg0: list["Points"]) -> None: ...

    def insert(self, i: int, x: Point) -> None:
        """Insert an item at a given position."""
        ...

    @abstractmethod
    def pop(self) -> Point:
        """Remove and return the last item"""
        ...

    @abstractmethod
    def pop(self, i: int) -> Point:
        """Remove and return the item at index i"""
        ...

    def remove(self, x: Point) -> None:
        """Remove the first item from the list whose value is x. It is an error if there is no such item."""
        ...

    def resize(self, arg0: int) -> None: ...


class Rectangle:
    """`dlib.rectangle`"""

    area: Callable[[], int]
    bl_corner: Callable[[], Point]
    bottom: Callable[[], int]
    br_corner: Callable[[], Point]
    center: Callable[[], Point]

    @abstractmethod
    def contains(self, point: Point) -> bool: ...
    @abstractmethod
    def contains(self, x: int, y: int) -> bool: ...
    @abstractmethod
    def contains(self, rectangle: "Rectangle") -> bool: ...

    dcenter: Callable[[], Point]
    height: Callable[[], int]

    @abstractmethod
    def intersect(self, rectangle: "Rectangle") -> "Rectangle": ...

    is_empty: Callable[[], bool]
    left: Callable[[], int]
    right: Callable[[], int]
    tl_corner: Callable[[], Point]
    top: Callable[[], int]
    tr_corner: Callable[[], Point]
    width: Callable[[], int]


class FullObjectDetection:
    """`dlib.full_object_detection`"""

    num_parts: int
    rect: Rectangle  # The bounding box of the object.

    @abstractmethod
    def part(self, idx: int) -> Point: ...

    @abstractmethod
    def parts(self) -> Points: ...


Detector = Callable[[NumPyNumericArray], list[Rectangle]]
Predictor = Callable[[NumPyNumericArray, Rectangle], FullObjectDetection]
