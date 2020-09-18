import os
import unittest

from embryovision import localfolders


class TestLocalFolders(unittest.TestCase):
    def test_embryovision_folder_stores_contains_file(self):
        contents = os.listdir(localfolders.embryovision_folder)
        should_be_there = ['localfolders.py', 'tests']
        for item in should_be_there:
            self.assertIn(item, contents)


if __name__ == '__main__':
    unittest.main()
