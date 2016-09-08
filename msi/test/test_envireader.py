# import unittest
#
#
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)
#
#
# if __name__ == '__main__':
#     unittest.main()

import unittest
from msi.io.envireader import EnviReader
import numpy as np


class TestEnviReader(unittest.TestCase):

    def setUp(self):
        self.enviReader = EnviReader()
        self.msi = self.enviReader.read('./msi/data/testMSIENVI')

    def test_read_does_not_crash(self):
        # if we got this far, at least an image was read.
        self.assertTrue(len(self.msi.get_image().shape) == 3,
                        "read image has correct basic shape dimensions")
        self.assertTrue(self.msi.get_image().shape[-1] == 16,
                        "read image has correct number of image stacks")
        # self.assertTrue(np.array_equal(self.msi.get_image()[2, 2, :],
        #                 np.array([1, 2, 3, 4, 5])),
        #                 "read image contains correct data")


