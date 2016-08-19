
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
DATA_PATH = os.path.join(this_dir, "..", "data")
PARAMS = ['sao2','a_mie','b_mie','g','n','d']

class TestTissueParser(unittest.TestCase):

    def setUp(self):
        self.tissue_instance = tissueparser.read_tissue_config(
            os.path.join(DATA_PATH,'tissue_config_test.ini'))

    def test_tissue_parser(self):
        self.assertEquals(len(self.tissue_instance), 4,
                          "Number of layers read is incorrect.")
        for iLayer in range(0,4,1):
            self.assertEquals(len(self.tissue_instance[iLayer].parameter_list), 6,
                          "Number of parameters read is incorrect for Layer-" +
                              str(iLayer))
            for iParameter in range(0, 5, 1):
                self.assertEquals(self.tissue_instance[iLayer].parameter_list[iParameter].name,
                                  PARAMS[iParameter],
                                  "Parameter: " +
                                  self.tissue_instance[iLayer].parameter_list[iParameter].name +
                                  " in Layer-" + str(iLayer) + " is incorrect.")


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()



