
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
import filecmp
import os

from mc.tissuemodels import GenericTissue

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "..", "data")


class TestTissueModels(unittest.TestCase):

    def setUp(self):
        self.mci_filename = "temp.mci"
        self.mco_filename_prefix = "temp_"
        self.correct_mci_filename = os.path.join(DATA_PATH, "colon_default.mci")

    def tearDown(self):
        os.remove(self.mci_filename)

    def test_tissue_model(self):
        # create nice colon model
        tissue = GenericTissue(nr_layers=2)
        tissue.set_mci_filename(self.mci_filename)
        tissue.set_base_mco_filename(self.mco_filename_prefix)
        wavelengths = [500. * 10 ** -9]
        # just use the default parameters for this test
        # now create the simulation file
        tissue.create_mci_file()
        tissue.update_mci_file(wavelengths)
        # and assert its correct
        self.assertTrue(os.path.isfile(self.mci_filename),
                        "mci file was created")
        self.assertTrue(filecmp.cmp(self.mci_filename,
                                    self.correct_mci_filename,
                                    shallow=False),
                "the written mci file is the same as the stored " +
                "reference file")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
