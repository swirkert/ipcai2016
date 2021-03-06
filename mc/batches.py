
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
Created on Oct 15, 2015

@author: wirkert
'''

import numpy as np
from pandas import DataFrame
import pandas as pd


class AbstractBatch(object):
    """summarizes a batch of simulated mc spectra"""

    def __init__(self):
        pass

    def _create_empty_batch(self):
        my_index = pd.MultiIndex(levels=[[], []],
                             labels=[[], []])
        df = DataFrame(columns=my_index)
        return df

    def create_tissue_samples(self, nr_samples):
        """create the parameters for the batch, the simulation has
        to create the resulting reflectances"""
        return self._create_empty_batch()

    def nr_elements(self):
        return self.df.shape[0]


class IniBatch(AbstractBatch):
    """n-layer batch configured by ini file """

    def __init__(self, tissue_instance=None):
        super(IniBatch, self).__init__()
        self.tissue_instance = tissue_instance

    def set_tissue_instance(self, tissue_instance):
        self.tissue_instance = tissue_instance

    def create_tissue_samples(self, nr_samples):
        """Create generic n layer batch. The parameters vary randomly
        within each layer according to the tissue_config"""

        df = self._create_empty_batch()

        for i, layer in enumerate(self.tissue_instance):

            for param in layer.parameter_list:
                if param.distribution == "uniform":
                    gen = _uniform_tissue_distribution
                elif param.distribution == "normal":
                    gen = _normal_tissue_distribution
                elif param.distribution == "step":
                    gen = _step_sample
                elif param.distribution == "same":
                    previous_layer_nr = i-1
                    variable_name = param.name
                    assert(previous_layer_nr >= 0)
                    previous_samples = df["layer" + str(previous_layer_nr)][variable_name]
                    gen = _FromPreviousLayer(previous_samples)
                else:
                    raise NotImplementedError("Unknown parameter distribution type")

                samples = gen(nr_samples, *param.values)
                df["layer" + str(i), param.name] = samples

        return df


# now the wrappers to the distributions
def _normal_tissue_distribution(nr_samples, mean, std):
    samples = np.random.normal(loc=mean, scale=std, size=nr_samples)
    # make sure they are > 0
    small_number = 10**-10
    samples = np.clip(samples, small_number, np.inf)
    return samples


def _uniform_tissue_distribution(nr_samples, low, high):
    samples = np.random.uniform(low, high, nr_samples)
    return samples


def _step_sample(nr_samples, start, stop):
    return np.linspace(start, stop, nr_samples)


class _FromPreviousLayer:
    # this has to be a functor to be able to use our interface. We sneak in the
    # non standard paramter from the previous layer in the constructor.

    def __init__(self, previous_samples):
        self.previous_samples = previous_samples

    def __call__(self, *args):
        return self.previous_samples


