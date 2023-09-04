import os
import unittest
from facer import facer


class TestEndpoints(unittest.TestCase):

    NUM_IMAGES = 4
    NUM_FACE_IMAGES = 3

    @classmethod
    def setUpClass(cls):
        print("\n---------------------\nSetting up Facer tests...\n")
        cls.root = "./tests/images"

    def test_image_file_globbing(self):
        msg = "Files names were not ingested properly."
        files = facer.glob_image_files(self.root)
        names = ["dawn-beatty.jpg", "github.jpg", "kyle-daigle.jpg", "thomas-dohmke.jpg"]
        expected = [os.path.join(self.root, name) for name in names]
        self.assertTrue(sorted(files) == sorted(expected), msg)

    def test_load_images(self):
        msg = "Incorrect number of images loaded."
        kwargs = {"verbose":True}
        images = facer.load_images(self.root, **kwargs)
        expected = self.NUM_IMAGES * [[]]
        self.assertTrue(len(images.keys()) == len(expected), msg)

    def test_load_face_landmarks(self):
        msg = "Couldn't load landmarks."
        landmarks = facer.load_face_landmarks(self.root)
        expected = self.NUM_FACE_IMAGES * [[]]
        self.assertTrue(len(landmarks) == len(expected))

    def test_detect_face_landmarks(self):
        msg = "Incorrect number of face landmarks found."
        kwargs = {"save_landmarks":True, "verbose":True, "max_faces":10}
        images = facer.load_images(self.root, verbose=False)
        landmarks, faces = facer.detect_face_landmarks(images, **kwargs)
        expected = self.NUM_FACE_IMAGES * [[]]
        self.assertTrue(len(landmarks) == len(expected), msg)

    # def test_create_average_face(self):
    #     msg = "Failed to create the average face."
    #     faces = self.faces
    #     landmarks = self.landmarks
    #     kwargs = {"save_image": False}
    #     average_face = facer.create_average_face(faces, landmarks, **kwargs)
    #     self.assertTrue(average_face != []) # Terrible test

    # def test_create_animated_gif(self):
    #     msg = "Failed to create an animated GIF."
    #     input_dir = "./tests/images"
    #     facer.test_create_animated_gif(input_dir)
