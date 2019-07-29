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
