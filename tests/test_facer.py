import os
import unittest
from facer import facer


class TestEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n---------------------\nSetting up Facer tests...\n")
        cls.root = "./tests/images"

    def test_image_file_globbing(self):
        msg = "Files names were not ingested properly."
        files = facer.glob_image_files(self.root)
        names = ["barack-obama.jpg", "jimmy-carter.png",
                 "ronald-reagan.jpeg", "logo.png", "black-sabbath.jpg"]
        expected = [os.path.join(self.root, name) for name in names]
        self.assertTrue(sorted(files) == sorted(expected), msg)

    def test_load_images(self):
        msg = "Incorrect number of images loaded."
        kwargs = {"verbose":True}
        images = facer.load_images(self.root, **kwargs)
        expected = 5 * [[]] # List of length 4 (kludge)
        self.assertTrue(len(images.keys()) == len(expected), msg)

    def test_load_face_landmarks(self):
        msg = "Couldn't load landmarks."
        landmarks = facer.load_face_landmarks(self.root)
        expected = 6 * [[]] # Kludge for now
        self.assertTrue(len(landmarks) == len(expected))

    def test_detect_face_landmarks(self):
        msg = "Incorrect number of face landmarks found."
        kwargs = {"save_landmarks":True, "verbose":True, "max_faces":10}
        images = facer.load_images(self.root, verbose=False)
        landmarks, faces = facer.detect_face_landmarks(images, **kwargs)
        expected = 6 * [[]] # List of length 3 (kludge)
        self.assertTrue(len(landmarks) == len(expected), msg)

    # def test_create_average_face(self):
    #     msg = "Failed to create the average face."
    #     faces = self.faces
    #     landmarks = self.landmarks
    #     kwargs = {"save_image": False}
    #     average_face = facer.create_average_face(faces, landmarks, **kwargs)
    #     self.assertTrue(average_face != []) # Terrible test
