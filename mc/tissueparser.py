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

import ConfigParser
import layers


def read_tissue_config(file_name):
    # Read the tissue configuration file
    tissue_instance = []

    tissue_config = ConfigParser.ConfigParser()
    tissue_config.read(file_name)

    # Create temporary variables

    # Iterate over all the layers in the configuration file
    for ilayer in tissue_config.sections():
        # Local 'Layer' instance
        tmp_layer = layers.Layer()
        tmp_layer.update_description(ilayer)
        # Iterate over all the parameters supplied for each layer
        for iparameter in tissue_config.options(ilayer):
            # Local 'LayerParam' instance
            tmp_layer_param = layers.LayerParam()
            tmp_layer_param.update_name(iparameter)

            # Get list of all the values and store them as float.
            # The last value is "distribution type" and stored as string.
            param_string = tissue_config.get(ilayer,iparameter)
            param_val_str = param_string.split(',')
            for ival in range(len(param_val_str)-1):
                tmp_layer_param.append_value(float(param_val_str[ival]))

            # Assuming that the last element is distribution type
            tmp_layer_param.update_desription(param_val_str[-1])
            tmp_layer_param.update_dist_type(param_val_str[-1])

            # To the temporary layer instance, add the parameter.
            tmp_layer.add_parameter(tmp_layer_param)

        # Add the newly created layer to the tissue instance.
        tissue_instance.append(tmp_layer)
        #tmp_layer = None

    if not tissue_instance:  # empty lists are fals
        raise IOError("could not find config file at " + file_name)

    return tissue_instance
