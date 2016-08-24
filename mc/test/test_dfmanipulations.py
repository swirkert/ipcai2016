
"""

ipcai2016

Copyright (c) German Cancer Research Center,
Computer Assisted Interventions.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE for details

"""
'''
Created on Oct 19, 2015

@author: wirkert
'''
import unittest
import os

import numpy as np
from pandas.util.testing import assert_frame_equal

import mc.dfmanipulations as dfmani
from mc.batches import IniBatch
from mc import tissueparser


this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "..", "data", "tissues")


class Test(unittest.TestCase):

    def setUp(self):
        # create a colon batch with 2 samples

        tissue_instance = tissueparser.read_tissue_config(
            os.path.join(DATA_PATH, 'tissue_config_test.ini'))
        self.test_batch = IniBatch(tissue_instance)
        self.df = self.test_batch.create_tissue_samples(2)

        # artificially add 10 fake "reflectances" to this batch
        # at 10 fake "wavelengths"
        WAVELENGHTS = np.linspace(450, 720, 10)
        reflectance1 = np.arange(0, 30, 3)
        reflectance2 = np.arange(30, 60, 3)
        for w in WAVELENGHTS:
            self.df["reflectances", w] = np.NAN
        for r1, r2, w in zip(reflectance1, reflectance2, WAVELENGHTS):
            self.df["reflectances", w][0] = r1
            self.df["reflectances", w][1] = r2

    def test_sliding_average(self):
        # by test design folding should not alter elements (only at boundaries,
        # which are excluded by array slicing:
        expected_elements = self.df.reflectances.iloc[:, 1:-1].copy()
        dfmani.fold_by_sliding_average(self.df, 3)

        assert_frame_equal(self.df.reflectances, expected_elements)

    def test_interpolation(self):
        new_wavelengths = [465, 615, 555]

        dfmani.interpolate_wavelengths(self.df, new_wavelengths)

        expected = np.array([[1.5, 16.5, 10.5], [31.5, 46.5, 40.5]])
        np.testing.assert_almost_equal(self.df.reflectances.as_matrix(),
                                       expected,
                                       err_msg="test if interpolation " +
                                       "works fine on batches")

    def test_select_n(self):
        """ this is less a test and more a showing of how to select n elements
            from a dataframe."""
        # draw one sample. Look into documentation for sample to see all the
        # options. Sample is quite powerfull.
        self.df = self.df.sample(1)
        self.assertEqual(self.df.shape[0], 1,
                         "one sample selected")

    def test_sortout_bands(self):
        """ this is less a test and more a showing of how to sortout specific
            bands from a dataframe """
        # drop the 510 and 720 nm band
        band_names_to_sortout = [510, 720]
        self.df.drop(band_names_to_sortout, axis=1, level=1, inplace=True)

        df_r = self.df["reflectances"]
        self.assertTrue(not (510 in df_r.columns))
        self.assertTrue(not 720 in df_r.columns)
        self.assertTrue(690 in df_r.columns)


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
