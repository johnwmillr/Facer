import os
import pytest
from facer import facer

NUM_IMAGES = 4
NUM_FACE_IMAGES = 3
ROOT = "./tests/images"


@pytest.fixture(scope="module")
def setup():
    print("\n---------------------\nSetting up Facer tests...\n")
    return ROOT


def test_image_file_globbing(setup):
    msg = "Files names were not ingested properly."
    files = facer.glob_image_files(setup)
    names = ["dawn-beatty.jpg", "github.jpg", "kyle-daigle.jpg", "thomas-dohmke.jpg"]
    expected = [os.path.join(setup, name) for name in names]
    assert sorted(files) == sorted(expected), msg


def test_load_images(setup):
    msg = "Incorrect number of images loaded."
    kwargs = {"verbose": True}
    images = facer.load_images(setup, **kwargs)
    expected = NUM_IMAGES * [[]]
    assert len(images.keys()) == len(expected), msg


def test_load_face_landmarks(setup):
    msg = "Couldn't load landmarks."
    landmarks = facer.load_face_landmarks(setup)
    expected = NUM_FACE_IMAGES * [[]]
    assert len(landmarks) == len(expected), msg


def test_detect_face_landmarks(setup):
    msg = "Incorrect number of face landmarks found."
    kwargs = {"save_landmarks": True, "verbose": True, "max_faces": 10}
    images = facer.load_images(setup, verbose=False)
    landmarks, faces = facer.detect_face_landmarks(images, **kwargs)
    expected = NUM_FACE_IMAGES * [[]]
    assert len(landmarks) == len(expected), msg


# def test_create_average_face(setup):
#     msg = "Failed to create the average face."
#     faces = self.faces
#     landmarks = self.landmarks
#     kwargs = {"save_image": False}
#     average_face = facer.create_average_face(faces, landmarks, **kwargs)
#     assert average_face != [], msg

# def test_create_animated_gif(setup):
#     msg = "Failed to create an animated GIF."
#     input_dir = "./tests/images"
#     facer.test_create_animated_gif(input_dir)
