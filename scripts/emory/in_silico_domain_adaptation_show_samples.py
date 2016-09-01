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
Created on Fri Aug 14 11:09:18 2015

@author: wirkert

Modified on August 16, 2016: Anant Vemuri
"""

import datetime
import logging
import os
from collections import namedtuple
import sys

import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.metrics import r2_score

import ipcai2016.tasks_mc as tasks_mc
from regression.preprocessing import preprocess, preprocess2, add_snr
from regression.linear import LinearSaO2Unmixing
from regression.da_sampling import sample

import commons


##########################################################

sc = commons.ScriptCommons()

sc.add_dir("IN_SILICO_RESULTS_PATH", os.path.join(sc.get_dir("RESULTS_FOLDER"),
                                     "in_silico"))

# emory universities camera full width at half maximum:
fwhm = 20*10**-9


font = {'family' : 'normal',
        'size'   : 5}

matplotlib.rc('font', **font)


# setup standard random forest
rf = RandomForestRegressor(10, min_samples_leaf=10, max_depth=9, n_jobs=-1)
EvaluationStruct = namedtuple("EvaluationStruct",
                              "name regressor")
# standard evaluation setup
standard_evaluation_setups = [EvaluationStruct("Linear Beer-Lambert",
                                               LinearSaO2Unmixing(wavelengths=sc.other["RECORDED_WAVELENGTHS"],
                                                                  fwhm=10*10**-9))
                              , EvaluationStruct("Proposed", rf)]

# an alternative if you want to compare non-linear to linear regression methods
# standard_evaluation_setups = [EvaluationStruct("Linear Regression",
#                                                LinearRegression())
#                               , EvaluationStruct("Proposed", rf)]

# color palette
my_colors = ["red", "green"]

# standard noise levels
noise_levels = np.array([5, 10, 50, 100]).astype("float")


class CreateSamples(luigi.Task):
    which_train = luigi.Parameter()
    which_validate = luigi.Parameter()
    physiological_parameters = luigi.Parameter()
    eval_name = luigi.Parameter()

    def requires(self):
        return tasks_mc.CameraBatch(self.which_train, fwhm, self.eval_name), \
               tasks_mc.CameraBatch(self.which_validate, fwhm, self.eval_name)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("IN_SILICO_RESULTS_PATH"),
                                              "foobar.png")) # this will always be executed

    def preprocess_paramter(self, batch, nr_samples=None, snr=None,
                            magnification=None, bands_to_sortout=None):
        """ For evaluating vhb we extract labels for vhb instead of sao2"""
        X, y = preprocess2(batch, nr_samples, snr,
                           magnification, bands_to_sortout, self.physiological_parameters)
        return X, y.values

    def run(self):
        data_folder = os.path.join(sc.get_full_dir("IN_SILICO_RESULTS_PATH"), "sample_fits")
        commons.create_folder_if_necessary(data_folder)

        # get data
        df_train = pd.read_csv(self.input()[0].path, header=[0, 1])
        df_validate = pd.read_csv(self.input()[1].path, header=[0, 1])
        nr_samples = 30

        for i in range(nr_samples):
            f, axarr = plt.subplots(len(noise_levels), sharey=True, sharex=True)
            for j, w in enumerate(noise_levels):
                X_validate, y_validate = self.preprocess_paramter(df_validate, snr=w)
                X_train, y_train = self.preprocess_paramter(df_train)
                # now get adapted dataset
                X_sampled, y_sampled = sample(X_train, y_train, X_validate[i],
                                              step_size=1, window_size=(1, 1))
                axarr[j].plot(sc.other["RECORDED_WAVELENGTHS"], X_sampled)
                axarr[j].plot(sc.other["RECORDED_WAVELENGTHS"], X_validate[i])

                title = _build_title(w, self.physiological_parameters, y_validate[i], y_sampled)
                axarr[j].set_title(title)

            plt.savefig(os.path.join(data_folder, str(i) + ".png"),
                        dpi=500)


def _build_title(noise, params, true, estimated):
    title = "SNR: " + str(noise) + " || "
    for p, t, e in zip(params, true, estimated):
        title += p + ": " + "{:.2f}".format(t) + "/" + "{:.2f}".format(e) + " || "
    return title


def main(args):
    eval_dict = commons.read_configuration_dict(args[1])

    eval_name = eval_dict["evaluation_name"]
    train = eval_dict["mc_data_train"]
    validate = eval_dict["mc_data_validate"]

    w_start = float(eval_dict["simulated_wavelengths_start"])
    w_end = float(eval_dict["simulated_wavelengths_stop"])
    w_step = float(eval_dict["simulated_wavelengths_step"])
    sc.other["RECORDED_WAVELENGTHS"] = np.arange(w_start, w_end, w_step) * 10 ** -9

    sc.set_root(eval_dict["root_path"])
    sc.create_folders()

    logging.basicConfig(filename=os.path.join(sc.get_full_dir("LOG_FOLDER"),
                                              eval_name + "_in_silico_da_sample_plots_" +
                                              str(datetime.datetime.now()) +
                                              '.log'),
                        level=logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(ch)
    luigi.interface.setup_interface_logging()

    sch = luigi.scheduler.CentralPlannerScheduler()
    w = luigi.worker.Worker(scheduler=sch)

    w.add(CreateSamples(which_train=train, which_validate=validate,
                        physiological_parameters=["vhb", "sao2", "a_mie", "a_ray"],
                        eval_name=eval_name))

    w.run()


if __name__ == '__main__':
    main(sys.argv)
