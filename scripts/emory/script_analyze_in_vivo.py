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
import sys
import pickle

import h5py
import luigi
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from msi.msi import Msi
from msi.io.nrrdwriter import NrrdWriter
from msi.io.nrrdreader import NrrdReader
import msi.msimanipulations as msimani
import msi.normalize as norm
import ipcai2016.tasks_mc as tasks_mc
import ipcai2016.tasks_common as tasks_common
from regression.estimation import estimate_np_image

import commons


##########################################################

sc = commons.ScriptCommons()

sc.add_dir("IN_VIVO_RESULTS_PATH", os.path.join(sc.get_dir("RESULTS_FOLDER"),
                                     "in_vivo"))

# emory universities camera full width at half maximum:
fwhm = 20*10**-9

font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)


class EvaluateEmoryImage(luigi.Task):
    which_train = luigi.Parameter()
    eval_name = luigi.Parameter()

    image = luigi.Parameter()
    white = luigi.Parameter()
    dark = luigi.Parameter()

    eval_dict = luigi.Parameter()

    def requires(self):
        return EmoryTrainRegressor(self.which_train, fwhm, self.eval_name), \
               CalibratedEmoryImage(self.image, self.white, self.dark, self.eval_dict)

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("IN_VIVO_RESULTS_PATH"),
                                              os.path.split(self.image)[1] +
                                              ".png"))

    def run(self):
        # get image
        nrrd_reader = NrrdReader()
        image = nrrd_reader.read(self.input()[1].path)

        # read the regressor
        e_file = open(self.input()[0].path, 'r')
        e = pickle.load(e_file)

        # zero values would lead to infinity logarithm, thus clip.
        image.set_image(np.clip(image.get_image().astype("float32"), 0.00001, 2.** 64))
        norm.standard_normalizer.normalize(image)
        # transform to absorption
        image.set_image(-np.log(image.get_image()))
        # normalize by l2 for stability
        norm.standard_normalizer.normalize(image, "l2")
        # estimate
        np_image, time = estimate_np_image(image, e)

        # plot parametric maps
        f, axarr = plt.subplots(np_image.shape[-1])
        plt.set_cmap("jet")
        # first oxygenation
        oxy_image = np_image[:, :, 0]
        oxy_image[np.isnan(oxy_image)] = 0.
        oxy_image[np.isinf(oxy_image)] = 0.
        oxy_mean = np.mean(oxy_image)
        oxy_image[0, 0] = 0.0
        oxy_image[0, 1] = 1.
        tasks_common.plot_image(oxy_image[:, :], axarr[0], cmap="jet", title="oxygenation")
        # second blood volume fraction
        vhb_image = np_image[:, :, 1]
        vhb_image[np.isnan(vhb_image)] = 0.
        vhb_image[np.isinf(vhb_image)] = 0.
        vhb_image[0, 0] = 0.0
        vhb_image[0, 1] = 0.1
        vhb_image = np.clip(vhb_image, 0.0, 0.1)
        vhb_mean = np.mean(vhb_image)
        tasks_common.plot_image(vhb_image, axarr[1], cmap="jet", title="vhb")

        # finally save the figure
        plt.savefig(self.output().path, dpi=500,
                    bbox_inches='tight')


class CalibratedEmoryImage(luigi.Task):
    image = luigi.Parameter()
    white = luigi.Parameter()
    dark = luigi.Parameter()
    eval_dict = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              "calibrated_" + os.path.split(self.image)[1] + ".nrrd"))

    def requires(self):
        return EmoryImage(self.image, self.eval_dict), \
               EmoryImage(self.white, self.eval_dict), \
               EmoryImage(self.dark, self.eval_dict)

    def run(self):
        nrrd_reader = NrrdReader()
        image = nrrd_reader.read(self.input()[0].path)
        white = nrrd_reader.read(self.input()[1].path)
        dark = nrrd_reader.read(self.input()[2].path)
        msimani.image_correction(image, white, dark)
        nrrd_writer = NrrdWriter(image)
        nrrd_writer.write(self.output().path)


class EmoryImage(luigi.Task):
    image = luigi.Parameter()
    eval_dict = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              os.path.split(self.image)[1] + ".nrrd"))

    def run(self):
        np_image = load_image(self.image)
        # only specified wavelengths. Note that they can only crop at the end
        w = self.eval_dict["RECORDED_WAVELENGTHS"]
        np_image_relevant = np_image[:,:,:len(w)]
        msi_image = Msi()
        msi_image.set_image(np_image_relevant, w)
        nrrd_writer = NrrdWriter(msi_image)
        nrrd_writer.write(self.output().path)


def load_image(image_name):
    hsi = h5py.File(image_name)
    image = hsi[hsi.keys()[0]][:].T
    return image


class EmoryTrainRegressor(luigi.Task):
    which_train = luigi.Parameter()
    fwhm = luigi.Parameter()
    eval_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("INTERMEDIATES_FOLDER"),
                                              "reg_emory_" +
                                              self.which_train + "_" + self.eval_name))

    def requires(self):
        return tasks_mc.CameraBatch(self.which_train, fwhm, self.eval_name)

    def run(self):
        tasks_common.train_regressor(self.input().path, self.output())


def main(args):
    eval_dict = commons.read_configuration_dict(args[1])

    eval_name = eval_dict["evaluation_name"]
    train = eval_dict["in_vivo_mc_data_train"]

    w_start = float(eval_dict["wavelengths_start"])
    w_end = float(eval_dict["wavelengths_stop"])
    w_step = float(eval_dict["wavelengths_step"])
    sc.other["RECORDED_WAVELENGTHS"] = np.arange(w_start, w_end, w_step) * 10 ** -9
    eval_dict["RECORDED_WAVELENGTHS"] = np.arange(w_start, w_end, w_step) * 10 ** -9

    sc.set_root(eval_dict["root_path"])
    sc.create_results_folders()

    logging.basicConfig(filename=os.path.join(sc.get_full_dir("LOG_FOLDER"),
                                              eval_name + "_in_vivo_plots_" +
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

    image_list = ["898", "897", "894", "892", "891", "890",  "888",  "885"]

    for i in image_list:
        base_loc = os.path.join(eval_dict["root_path"], "data", i)
        image_loc = os.path.join(base_loc, "HSI", "im_" + i + ".mat")
        white_image_loc = os.path.join(base_loc, "Reference", "white_" + i + ".mat")
        dark_image_loc = os.path.join(base_loc, "Reference", "dark_" + i + ".mat")

        eval_image_task = EvaluateEmoryImage(train, eval_name,
                                             image_loc, white_image_loc, dark_image_loc,
                                             eval_dict)
        w.add(eval_image_task)

    w.run()


if __name__ == '__main__':
    main(sys.argv)
