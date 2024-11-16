from typing import Any

# Convenience typing for the untyped dlib code
Detector = Any
Predictor = Any
DlibImage = Any


class PointT:
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y
