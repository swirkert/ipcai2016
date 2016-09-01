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

import numpy as np


def sample(X_s, y_s, X_t, step_size=5, window_size=(20, 20),  local_cov=False):
    """
    Samples from X_s based on similarities to the multispectral image X_t

    n: nr_samples
    m: nr_features

    Args:
        X_s: n1xm
        y_s: n1 labels for X_s
        X_t: n21xn22xm

    Returns: (X_resampled, y_resampled) : (n21xn22xm resampled multi spectral image, n21xn22 dimensional labels)
    """
    # we always want to work with "an image"
    if len(X_t.shape) == 2:
        X_t = X_t.reshape((X_t.shape[0], 1, -1))
    if len(y_s.shape) == 1:
        y_s = y_s.reshape((y_s.shape[0], 1))

    sampled_array = np.zeros(X_t.shape)
    y_resampled = np.zeros((X_t.shape[0], X_t.shape[1], y_s.shape[-1]), dtype=y_s.dtype)
    choice_indices = range(X_s.shape[0])
    inv_cov_calculator = InvCovCalculator(X_t, local_cov)

    for i, j, center, window in sliding_window(X_t, step_size, window_size):
        inv_cov_estimate = inv_cov_calculator.determine_inv_cov(window)
        # determine similarity
        X_s_centered = X_s - center
        p_maha = distance_to_probablility(d_maha)

        # draw elements and add to results
        max_idx = np.argmax(p_maha)
        sampled_array[i, j, :] = X_s[max_idx, :]
        y_resampled[i, j, :] = y_s[max_idx, :]
        print i, j

    return np.squeeze(sampled_array[:i+1, :j+1, :]), np.squeeze(y_resampled[:i+1, :j+1, :])


class InvCovCalculator:

    def __init__(self, X, local_cov=False):
        self.X = X
        self.local_cov = local_cov
        self.global_invcov = _det_inv_cov(X)

    def determine_inv_cov(self, X_local):
        if not self.local_cov:
            return self.global_invcov
        else:
            return _det_inv_cov(X_local)


def _det_inv_cov(X):
    X_flattend = X.reshape((X.shape[0]*X.shape[1], -1))  # n21*n22xm
    return np.linalg.inv(np.cov(X_flattend.T))


def distance_to_probablility(d_maha):
    d_maha = np.squeeze(d_maha)
    # use of divide will set division by zero values to 0
    inv_d_maha = np.divide(1., d_maha)
    # now calculate probablility
    e_inv_d_maha = np.exp(inv_d_maha)  # tried to do it with softmax
    return e_inv_d_maha / np.sum(e_inv_d_maha)


# adapted from
# http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def sliding_window(image, stepSize, windowSize):
    half_x_window = windowSize[0]/2
    half_y_window = windowSize[1]/2
    # slide a window across the image
    for i, x in enumerate(xrange(half_x_window, image.shape[0]-half_x_window, stepSize)):
        for j, y in enumerate(xrange(half_y_window, image.shape[1]-half_y_window, stepSize)):
            # yield the current window
            center = image[x, y, :]
            yield (i, j, center, image[x - half_x_window:x + half_x_window,
                                       y - half_y_window:y + half_y_window, :])




