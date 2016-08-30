'''
Created on Oct 19, 2015

@author: wirkert
'''
import unittest

import numpy as np
import mc.camera as cam


class TestCamera(unittest.TestCase):

    def setUp(self):
        # setup a imaging system that does almost nothing
        # (camera with 8 bands, each the same
        # and just taking all the wavelengths into account equally).
        # The light source is completely homogeneous w.r.t. wavelength
        self.nr_bands = 8
        self.nr_wavelengths = 200
        wavelengths = np.linspace(400, 700, self.nr_wavelengths) * 10**-9  # nm
        filters = np.ones((self.nr_bands, self.nr_wavelengths))
        self.imaging_system = cam.ImagingSystem(wavelengths=wavelengths,
                                                F=filters)


    def test_reflectance_transform(self):
        reflectance = 0.5 * np.ones(self.nr_wavelengths)
        # we expect a normalized result with all values equal.
        expected_measurement = np.ones(self.nr_bands) / self.nr_bands
        calc_cam_measurement = cam.transform_reflectance(self.imaging_system,
                                                         reflectance)
        np.testing.assert_almost_equal(np.squeeze(calc_cam_measurement),
                                       np.squeeze(expected_measurement),
                                       err_msg="test if " +
                                               "camera measurement is " +
                                               "correctly calculated from " +
                                               "given reflectance")

    def test_spectrometer_transform(self):
        # measurement made by a spectrometer:
        spectrometer = 0.5 * np.ones(self.nr_wavelengths)
        # we expect a normalized result with all values equal.
        expected_measurement = np.ones(self.nr_bands) / self.nr_bands
        calc_cam_measurement = cam.transform_color(self.imaging_system,
                                                         spectrometer)
        np.testing.assert_almost_equal(np.squeeze(calc_cam_measurement),
                                       np.squeeze(expected_measurement),
                                       err_msg="test if " +
                                               "camera measurement is " +
                                               "correctly calculated from " +
                                               "given spectrometer measurement")

    def test_difference_transformations(self):
        # first set a non uniform lighting
        self.imaging_system.w = _n_random_numbers(self.nr_wavelengths)
        # also, let one filter differ from the rest
        self.imaging_system.F[0, :] = _n_random_numbers(self.nr_wavelengths)
        measurement = 0.5 * np.ones(self.nr_wavelengths)

        r_transformed = cam.transform_color(self.imaging_system, measurement)
        s_transformed = cam.transform_reflectance(self. imaging_system, measurement)

        # the two vectors should be different because one transformation is
        # taking lighting into account and the other one is not.
        self.assertFalse(r_transformed[0] == s_transformed[0], "test if " +
                         "color and reflectance transformations differ")


def _n_random_numbers(n):
    return np.array([np.random.uniform(0.1, 1) for _ in xrange(n)])

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
