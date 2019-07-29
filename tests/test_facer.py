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
        expected = [os.path.join(self.root, "barack-obama.jpg"),
                    os.path.join(self.root, "jimmy-carter.png"),
                    os.path.join(self.root, "ronald-regan.jpeg")]
        self.assertTrue(files == expected, msg)

    def test_detect_face_landmarks(self):
        msg = "Incorrect number of face landmarks found."
        kwargs = {"save_landmarks":False, "verbose":False}
        landmarks = facer.detect_face_landmarks(self.root, **kwargs)
        expected = [[], [], []] # List of length 3 (kludge)
        self.assertTrue(len(landmarks) == len(expected), msg)
