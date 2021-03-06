
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
Created on Oct 23, 2015

@author: wirkert
'''
import unittest

from mc.usuag import UsgJacques


class TestUs(unittest.TestCase):

    def setUp(self):
        self.usg = UsgJacques()

    def test_no_rayleigh_high_wavelengths(self):
        self.usg.a_ray = 2.*100
        self.usg.a_mie = 20.*100
        w = 500. * 10 ** -9
        print self.usg(w)[0] / 100.
        # todo write test


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
