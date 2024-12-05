import os

import pytest

from facer.facer import (
    constrainPoint,
    detect_face_landmarks,
    glob_image_files,
    load_face_landmarks,
    load_images,
)
from facer.utils import rectContains

NUM_IMAGES = 4
NUM_FACE_IMAGES = 3
ROOT = "./tests/images"


@pytest.fixture(scope="module")
def setup():
    print("\n---------------------\nSetting up Facer tests...\n")
    return ROOT


def test_image_file_globbing(setup: str) -> None:
    msg = "Files names were not ingested properly."
    files = glob_image_files(setup)
    names = ["dawn-beatty.jpg", "github.jpg", "kyle-daigle.jpg", "thomas-dohmke.jpg"]
    expected = [os.path.join(setup, name) for name in names]
    assert sorted(files) == sorted(expected), msg


def test_load_images(setup: str) -> None:
    msg = "Incorrect number of images loaded."
    images = load_images(setup, verbose=False)
    assert len(images.keys()) == NUM_IMAGES, msg


def test_load_face_landmarks(setup: str) -> None:
    msg = "Couldn't load landmarks."
    landmarks = load_face_landmarks(setup)
    assert len(landmarks) == NUM_FACE_IMAGES, msg


def test_detect_face_landmarks(setup: str) -> None:
    msg = "Incorrect number of face landmarks found."
    images = load_images(setup, verbose=False)
    landmarks, _ = detect_face_landmarks(
        images, save_landmarks=True, verbose=False, max_faces=10
    )
    assert len(landmarks) == NUM_FACE_IMAGES, msg


@pytest.mark.parametrize(
    ["point", "is_contained"],
    [[(0, 0), True], [(50, 50), True], [(100, 100), True], [(101, 101), False]],
)
def test_rectContains(point: tuple[float, float], is_contained: bool) -> None:
    rect = (0, 0, 100, 100)
    assert rectContains(rect, point) == is_contained


@pytest.mark.parametrize(
    ["point", "w", "h", "constrained_point"],
    [
        [(0, 0), 100, 100, (0, 0)],
        [(50, 50), 100, 100, (50, 50)],
        [(100, 100), 100, 100, (99, 99)],
        [(101, 101), 100, 100, (99, 99)],
    ],
)
def test_constrainPoint(
    point: tuple[int, int], w: int, h: int, constrained_point: tuple[int, int]
) -> None:
    assert constrainPoint(point, w, h) == constrained_point
