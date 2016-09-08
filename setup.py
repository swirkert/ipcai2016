# -*- coding: utf-8 -*-
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

"""
Created on Fri Aug  7 18:41:50 2015

@author: wirkert
"""

from setuptools import setup, find_packages

setup(name='ipcai2016',
      version='0.1',
      description='The code for the ipcai2016 submission. This toolkit is created during the pursuit of my phd to enable more easy processing of medical multispectral imaging data.',
      author='Sebastian Wirkert',
      author_email='s.wirkert@dkfz-heidelberg.de',
      license='BSD',
      packages=find_packages(exclude=['test*']),
      package_dir={},
      package_data={'data': ['*.txt', '*.mci', '*.nrrd']},
      install_requires=['numpy>=1.10.2', 'scipy', 'scikit-learn>=0.17',
                        'SimpleITK>=0.9.0', 'subprocess32',
                        'pypng', 'pandas>0.17', 'libtiff', 'Pillow', 'spectral'],
      entry_points={}  # for scripts, add later
      )
