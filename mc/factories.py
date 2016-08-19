
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

from mc.tissuemodels import AbstractTissue, GenericTissue
from mc.batches import AbstractBatch, IniBatch


class AbstractMcFactory(object):
    '''
    Monte Carlo Factory.
    Will create fitting models and batches, dependent on your task
    '''

    def create_tissue_model(self):
        return AbstractTissue()

    def create_batch_to_simulate(self):
        return AbstractBatch()

    def __init__(self):
        '''
        Constructor
        '''


class GenericMcFactory(AbstractMcFactory):

    def create_tissue_model(self):
        return GenericTissue()

    def create_batch_to_simulate(self):
        return IniBatch()

    def __init__(self):
        '''
        Constructor
        '''
