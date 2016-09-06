
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


def get_mcml_path():

    # Read the tissue configuration file
    # MODIFY THIS PATH FOR YOUR INSTALLATION
    mcml_exec_path = ''
    try:
        if not mcml_exec_path:
            raise ValueError('ERROR: Empty path. Please modify the path in mcmlpath.py.')
    except ValueError as e:
        print(e)
    return mcml_exec_path
