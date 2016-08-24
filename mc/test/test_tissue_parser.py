
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

from mc import tissueparser

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "..", "data", "tissues")
PARAMS = ['sao2', 'a_mie', 'b_mie', 'g', 'n', 'd']


class TestTissueParser(unittest.TestCase):

    def setUp(self):
        self.tissue_instance = tissueparser.read_tissue_config(
            os.path.join(DATA_PATH, 'tissue_config_test.ini'))

    def test_tissue_parser(self):
        self.assertEquals(len(self.tissue_instance), 4,
                          "Number of layers read is incorrect.")
        for i, layer in enumerate(self.tissue_instance):
            self.assertEquals(len(layer.parameter_list), 6,
                          "Number of parameters read is incorrect for Layer-" +
                              str(i))
            for desired_parameter, read_parameter in zip(PARAMS, layer.parameter_list):
                self.assertEquals(read_parameter.name,
                                  desired_parameter,
                                  "Parameter: " +
                                  read_parameter.name +
                                  " in Layer-" + str(i) + " is incorrect.")

    def test_tissue_parser_wrong_filename(self):
        with self.assertRaises(IOError):
            tissueparser.read_tissue_config("fakefiledoesnotexists.ini")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



