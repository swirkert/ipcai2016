'''
Created on Oct 19, 2015

@author: wirkert
'''
import unittest

import numpy as np

from regression.da_sampling import sample


class TestDaSampling(unittest.TestCase):

    def setUp(self):
        self.X_s = np.array([[0.5, 0.3, 0.5], [2, 0, -2], [0, 1, 0]], dtype=float)
        self.y_s = np.array([0, 1, 2])
        self.X_t = np.random.random_sample((1000, 3))
        self.X2d_t = np.random.random_sample((500, 500, 3))

    def test_1d_sampling(self):
        X_st, y_st = sample(self.X_s, self.y_s, self.X_t, step_size=1, window_size=(1, 1))
        self.assertEquals(X_st.shape, self.X_t.shape)
        np.testing.assert_allclose(X_st, self.X_s[y_st])

    def test_1d_sampling_stride(self):
        X_st, y_st = sample(self.X_s, self.y_s, self.X_t, step_size=5, window_size=(1, 1))
        self.assertEquals(X_st.shape, (self.X_t.shape[0]/5, self.X_t.shape[1]))

    def test_2d_sampling(self):
        X_st, y_st = sample(self.X_s, self.y_s, self.X2d_t, step_size=5, window_size=(1, 1))
        self.assertEquals(X_st.shape, (self.X2d_t.shape[0]/5,
                                       self.X2d_t.shape[1]/5,
                                       self.X2d_t.shape[2]))

    def test_2d_sampling_window_local_cov(self):
        # hard to test, this basically just checks if it doesn't crash
        # and returns the correct sizes
        X_st, y_st = sample(self.X_s, self.y_s, self.X2d_t, step_size=5, window_size=(20, 20),
                            local_cov=True)
        self.assertEquals(X_st.shape, (self.X2d_t.shape[0]/5-4,
                                       self.X2d_t.shape[1]/5-4,
                                       self.X2d_t.shape[2]))
        np.testing.assert_allclose(X_st, self.X_s[y_st])


