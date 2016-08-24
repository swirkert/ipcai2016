
"""

Copyright (c) German Cancer Research Center,
Computer Assisted Interventions.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE for details

"""
'''
Created on Aug 19, 2016

@author: avemuri
'''

import unittest
import os

import numpy as np

from mc import tissueparser
from mc.batches import IniBatch

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "..", "data", "tissues")


def is_between(array, min, max):
    larger_min = np.min(array) > min
    lower_max = np.max(array) < max
    return larger_min and lower_max


class TestIniBatch(unittest.TestCase):

    def setUp(self):
        self.tissue_instance = tissueparser.read_tissue_config(
            os.path.join(DATA_PATH, 'laparoscopic_ipcai_colon_2016_08_23.ini'))
        self.ini_batch = IniBatch()
        self.ini_batch.set_tissue_instance(self.tissue_instance)

    def test_ini_batch(self):
        df = self.ini_batch.create_tissue_samples(10)
        np.testing.assert_allclose(df["layer0"]["sao2"], df["layer1"]["sao2"])
        np.testing.assert_allclose(df["layer1"]["sao2"], df["layer2"]["sao2"])
        self.assertTrue(is_between(df["layer0"]["sao2"], 0.0, 1.))
        self.assertTrue(is_between(df["layer0"]["vhb"], 0.0, 0.1))
        np.testing.assert_allclose(df["layer0"]["b_mie"], 1.286)
        np.testing.assert_allclose(df["layer2"]["b_mie"], 1.286)
        self.assertTrue(is_between(df["layer1"]["g"], 0.8, 0.95))
        np.testing.assert_allclose(df["layer0"]["n"], 1.36)
        np.testing.assert_allclose(df["layer1"]["n"], 1.36)
        np.testing.assert_allclose(df["layer2"]["n"], 1.38)
        self.assertTrue(is_between(df["layer0"]["d"], 0.0006, 0.00101))
        self.assertTrue(is_between(df["layer1"]["d"], 0.000415, 0.000847))
        self.assertTrue(is_between(df["layer2"]["d"], 0.000395, 0.000603))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



