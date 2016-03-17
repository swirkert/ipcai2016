# ipcai2016
This software was created for the IPCAI 2016 submission:
"Robust Near Real-Time Estimation of Physiological Parameters from Megapixel
Multispectral Images with Inverse Monte Carlo and Random Forest Regression"

It features a small library to read and process multispectral images and a python wrapper for multi-layered Monte Carlo (MCML) models.

MCML:
http://omlc.org/software/mc/mcml/MCman.pdf
https://code.google.com/archive/p/gpumcml/

# installation
To install, navigate to this folder and type
python setup.py develop

The install requires can be found in the setup.py

## test
to run the unit tests execute
python -m unittest discover
in the folder where this file resides

# structure
The package is comprised of:

## folders: mc, msi, regression
these are packages with functionalities useful for multispectral imaging.

###mc
to do Monte Carlo (MC) simulations for multispectral imaging. Assumes a version of the MCML software is available somewhere on the system (also see tutorials)
###msi
to work with multispectral image stacks (read, write, normalize, plot, ...)
###regression
to apply machine learning regression on multsispectral images. This is under development

## folder scripts
here lie scritps which actually execute code, as e.g. the code for my ipcai 2016 submission:
"Robust Near Real-Time Estimation of Physiological Parameters from Megapixel
Multispectral Images with Inverse Monte Carlo and Random Forest Regression"

## folder tutorials
here lie the tutorials as ipython notebooks. Currently, only one examplary tutorial is finished, more to come.
