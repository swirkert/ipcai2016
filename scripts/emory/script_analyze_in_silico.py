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
from regression.preprocessing import preprocess, preprocess2
from regression.linear import LinearSaO2Unmixing

import commons


##########################################################

sc = commons.ScriptCommons()

sc.add_dir("IN_SILICO_RESULTS_PATH", os.path.join(sc.get_dir("RESULTS_FOLDER"),
                                     "in_silico"))

# emory universities camera full width at half maximum:
fwhm = 20*10**-9

w_standard = 10.  # for this evaluation we add 10% noise

font = {'family' : 'normal',
        'size'   : 20}

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
noise_levels = np.array([5, 10, 50, 100, 200]).astype("float")


class PhysiologicalParameterPlot(luigi.Task):
    which_train = luigi.Parameter()
    which_test = luigi.Parameter()
    train_snr = luigi.Parameter()
    physiological_parameter = luigi.Parameter()
    eval_name = luigi.Parameter()

    def requires(self):
        return tasks_mc.CameraBatch(self.which_train, fwhm, self.eval_name), \
               tasks_mc.CameraBatch(self.which_test, fwhm, self.eval_name)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("IN_SILICO_RESULTS_PATH"),
                                              self.eval_name + "_" +
                                              self.physiological_parameter +
                                              "_noise_plot_train_" +
                                              self.which_train + "_" +
                                              str(self.train_snr) + "SNR_"
                                              "_test_" + self.which_test +
                                              ".png"))

    def preprocess_paramter(self, batch, nr_samples=None, snr=None,
                            magnification=None, bands_to_sortout=None):
        """ For evaluating vhb we extract labels for vhb instead of sao2"""
        X, y = preprocess2(batch, nr_samples, snr,
                           magnification, bands_to_sortout, self.physiological_parameter)

        return X, y.values

    def run(self):
        # get data
        df_train = pd.read_csv(self.input()[0].path, header=[0, 1])
        df_test = pd.read_csv(self.input()[1].path, header=[0, 1])

        # for vhb we only evaluate the proposed method since the linear
        # beer-lambert is not applicable
        evaluation_setups = [EvaluationStruct("Proposed", rf)]
        df = evaluate_data(df_train, np.ones_like(noise_levels) * self.train_snr,
                           df_test, noise_levels,
                           evaluation_setups=evaluation_setups,
                           preprocessing=self.preprocess_paramter)
        standard_plotting(df, color_palette=["green"],
                          xytext_position=(2, 3), parameter_name=self.physiological_parameter)

        # finally save the figure
        plt.savefig(self.output().path, dpi=500,
                    bbox_inches='tight')


def evaluate_data(df_train, w_train, df_test, w_test,
                  evaluation_setups=None, preprocessing=None):
    """ Our standard method to evaluate the data. It will fill a DataFrame df
    which saves the errors for each evaluated setup"""
    if evaluation_setups is None:
        evaluation_setups = standard_evaluation_setups
    if preprocessing is None:
        preprocessing = preprocess
    if ("weights" in df_train) and df_train["weights"].size > 0:
        weights = df_train["weights"].as_matrix().squeeze()
    else:
        weights = np.ones(df_train.shape[0])

    # create a new dataframe which will hold all the generated errors
    df = pd.DataFrame()
    for one_w_train, one_w_test in zip(w_train, w_test):
        # setup testing function
        X_test, y_test = preprocessing(df_test, snr=one_w_test)
        # extract noisy data
        X_train, y_train = preprocessing(df_train, snr=one_w_train)
        for e in evaluation_setups:
            regressor = e.regressor
            regressor.fit(X_train, y_train, weights)
            y_pred = regressor.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            # save results to a dataframe
            errors = np.abs(y_pred - y_test)
            errors = errors.reshape(len(errors), 1)
            current_df = DataFrame(errors,
                                   columns=["absolute error"])
            current_df["Method"] = e.name
            current_df["SNR"] = int(one_w_test)
            current_df["r2"] = r2
            df = pd.concat([df, current_df], ignore_index=True)

    return df


def standard_plotting(df, color_palette=None, xytext_position=None,
                      parameter_name=""):
    if color_palette is None:
        color_palette = my_colors
    if xytext_position is None:
        xytext_position = (2, 15)

    plt.figure()

    # group it by method and noise level and get description on the errors
    df_statistics = df.groupby(['Method', 'SNR']).describe()
    # get the error description in the rows:
    df_statistics = df_statistics.unstack(-1)
    # get rid of multiindex by dropping "Error" level
    df_statistics = df_statistics["absolute error"]

    # iterate over methods to plot linegraphs with error tube
    # probably this can be done nicer, but no idea how exactly

    for color, method in zip(
            color_palette, df_statistics.index.get_level_values("Method").unique()):
        df_method = df_statistics.loc[method]
        plt.plot(df_method.index, df_method["50%"],
                 color=color, label=method)
        plt.fill_between(df_method.index, df_method["25%"], df_method["75%"],
                         facecolor=color, edgecolor=color,
                         alpha=0.5)
    # tidy up the plot
    plt.gca().set_xticks(noise_levels, minor=True)
    mean_r2 = df["r2"].mean()
    plt.xlabel("SNR")
    plt.ylabel("absolute error [%]")
    plt.title("evaluating " + parameter_name + ". Mean r2: " + str(mean_r2))
    plt.grid()
    plt.legend()


def main(args):
    eval_dict = commons.read_configuration_dict(args[1])

    eval_name = eval_dict["evaluation_name"]
    train = eval_dict["mc_data_train"]
    test = eval_dict["mc_data_test"]

    w_start = float(eval_dict["simulated_wavelengths_start"])
    w_end = float(eval_dict["simulated_wavelengths_stop"])
    w_step = float(eval_dict["simulated_wavelengths_step"])
    sc.other["RECORDED_WAVELENGTHS"] = np.arange(w_start, w_end, w_step) * 10 ** -9

    sc.set_root(eval_dict["root_path"])
    sc.create_folders()

    logging.basicConfig(filename=os.path.join(sc.get_full_dir("LOG_FOLDER"),
                                              eval_name + "_in_silico_plots_" +
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

    w.add(PhysiologicalParameterPlot(which_train=train, which_test=test,
                                     train_snr=10., physiological_parameter="sao2",
                                     eval_name=eval_name))

    w.add(PhysiologicalParameterPlot(which_train=train, which_test=test,
                                     train_snr=10., physiological_parameter="vhb",
                                     eval_name=eval_name))

    w.add(PhysiologicalParameterPlot(which_train=train, which_test=test,
                                     train_snr=10., physiological_parameter="a_mie",
                                     eval_name=eval_name))

    w.add(PhysiologicalParameterPlot(which_train=train, which_test=test,
                                     train_snr=10., physiological_parameter="a_ray",
                                     eval_name=eval_name))

    w.add(PhysiologicalParameterPlot(which_train=train, which_test=test,
                                     train_snr=10., physiological_parameter="d",
                                     eval_name=eval_name))
    w.run()


if __name__ == '__main__':
    main(sys.argv)
