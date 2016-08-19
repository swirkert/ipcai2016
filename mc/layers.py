
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


class LayerParam:
    name = None
    # Typical values include [min,max] for Uniform distribution
    # OR
    # [mean,variance] for Normal distribution

    def __init__(self, name=None, value_list=None, dist_type=None,
                 description=None):

        if name is  None:
            name = ''
        if value_list is  None:
            value_list = []
        if dist_type is  None:
            dist_type = ''
        if description is  None:
            description = "None Provided"

        self.name = name
        self.values = value_list
        self.distribution = dist_type
        self.description = description

    def update_name(self, name):

        self.name = name

    def update_values(self, value_list):

        self.values = value_list

    def update_dist_type(self, dist_type):
        # Possible values
        # 'normal', 'uniform', 'same'
        self.distribution = dist_type

    def update_desription(self, description=""):

        self.description = description

    def append_value(self, value):

        self.values.append(value)



class Layer:

    def __init__(self, parameter_list=None, description=None):
        # type: (list, string) -> Layer instance

        if parameter_list is  None:
            parameter_list = []
        if description is  None:
            description = "None Provided"

        self.parameter_list = parameter_list
        self.description = description

    def add_parameter(self, layer_param):

        self.parameter_list.append(layer_param)

    def update_description(self, description):

        self.description = description
