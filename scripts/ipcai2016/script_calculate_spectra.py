
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
Created on Sep 9, 2015

@author: wirkert

Modified on August 8, 2016: Anant Vemuri
'''

import datetime
import logging
import os
import time
import sys
import copy

import luigi
import mc.factories as mcfac
import numpy as np
from mc.sim import SimWrapper, get_diffuse_reflectance
from mc import tissueparser

from mc.mcmlpath import get_mcml_path

import commons

# parameter setting
NR_BATCHES = 2
NR_ELEMENTS_IN_BATCH = 10
# the wavelengths to be simulated
WAVELENGHTS = np.arange(450, 978, 20) * 10 ** -9
NR_PHOTONS = 10 ** 6

# experiment configuration
# this path definitly needs to be adapted by you
PATH_TO_MCML = get_mcml_path()

EXEC_MCML = "gpumcml.sm_20"
OUTPUT_ROOT_PATH = "/media/wirkert/data/Data/temp"

sc = commons.ScriptCommons()


class CreateSpectraTask(luigi.Task):
    tissue_file = luigi.Parameter()
    batch_nr = luigi.IntParameter()
    nr_samples = luigi.IntParameter()
    factory = luigi.Parameter()

    def output(self):
        path, file = os.path.split(self.tissue_file)
        df_prefix = os.path.splitext(file)[0]

        return luigi.LocalTarget(os.path.join(sc.get_full_dir("MC_DATA_FOLDER"),
                                              df_prefix,
                                              str(self.batch_nr) + ".csv"))

    def run(self):
        start = time.time()

        # determine df_prefix. Note code duplication in output method
        # #FIX: avoid this by finding out how to amend something to the
        # luigi constructor
        path, file = os.path.split(self.tissue_file)
        df_prefix = os.path.splitext(file)[0]

        # create folder for mci files if not exists
        mci_folder = os.path.join(sc.get_full_dir("MC_DATA_FOLDER"), df_prefix,
                                  "mci")
        if not os.path.exists(mci_folder):
            os.makedirs(mci_folder)

        # Setup simulation wrapper
        sim_wrapper = SimWrapper()
        sim_wrapper.set_mcml_executable(os.path.join(PATH_TO_MCML, EXEC_MCML))
        sim_wrapper.set_mci_filename(os.path.join(mci_folder, "Bat_" + str(self.batch_nr) + ".mci"))

        # Setup tissue model
        tissue_model = self.factory.create_tissue_model()
        tissue_model.set_mci_filename(sim_wrapper.mci_filename)
        tissue_model.set_nr_photons(NR_PHOTONS)
        tissue_model._mci_wrapper.set_nr_runs(
            NR_ELEMENTS_IN_BATCH * WAVELENGHTS.shape[0])
        tissue_model.create_mci_file()

        tissue_instance = tissueparser.read_tissue_config(self.tissue_file)

        # setup array in which data shall be stored
        batch = self.factory.create_batch_to_simulate()
        batch.set_tissue_instance(tissue_instance)

        # create the tissue samples and return them in dataframe df
        df = batch.create_tissue_samples(self.nr_samples)
        # add reflectance column to dataframe
        for w in WAVELENGHTS:
            df["reflectances", w] = np.NAN

        # Generate MCI file which contains list of all simulations in a Batch
        for i in range(df.shape[0]):
            # set the desired element in the dataframe to be simulated
            base_mco_filename = _create_mco_filename_for(df_prefix,
                                                         self.batch_nr,
                                                         i)
            tissue_model.set_base_mco_filename(base_mco_filename)
            tissue_model.set_tissue_instance(df.loc[i, :])
            tissue_model.update_mci_file(WAVELENGHTS)

        # Run simulations for computing reflectance from parameters
        sim_wrapper.run_simulation()

        # get information from created mco files
        for i in range(df.shape[0]):
            for wavelength in WAVELENGHTS:
                # for simulation get which mco file was created
                simulation_path = os.path.split(sim_wrapper.mcml_executable)[0]
                base_mco_filename = _create_mco_filename_for(df_prefix,
                                                             self.batch_nr,
                                                             i)
                mco_filename = base_mco_filename + str(wavelength) + '.mco'
                # get diffuse reflectance from simulation
                df["reflectances", wavelength][i] = \
                    get_diffuse_reflectance(os.path.join(simulation_path, mco_filename))
                # delete created mco file
                os.remove(os.path.join(simulation_path, mco_filename))

        f = open(self.output().path, 'w')
        df.to_csv(f)

        end = time.time()
        logging.info("time for creating batch of mc data: %.f s" %
                     (end - start))


def _create_mco_filename_for(prefix, batch, simulation):
    return str(prefix) + "_Bat_" + str(batch) + "_Sim_" + str(simulation) + "_"


def main(args):
    # create a folder for the results if necessary
    sc.set_root(OUTPUT_ROOT_PATH)
    sc.create_folders()

    logging.basicConfig(filename=os.path.join(sc.get_full_dir("LOG_FOLDER"),
                                 "calculate_spectra" +
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
    BATCH_NUMBERS = np.arange(0, NR_BATCHES, 1)
    for i in BATCH_NUMBERS:
        task = CreateSpectraTask(args[1],
                                 i,
                                 NR_ELEMENTS_IN_BATCH,
                                 mcfac.GenericMcFactory())
        w.add(task)
        w.run()


if __name__ == '__main__':

    args = copy.deepcopy(sys.argv)
    if len(sys.argv) == 1:
        args.append("/home/wirkert/workspace/ipcai2016_new/mc/data/tissues/laparoscopic_ipcai_colon_2016_08_23.ini")

    main(args)
