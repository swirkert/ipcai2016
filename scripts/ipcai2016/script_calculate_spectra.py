
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
import ConfigParser

import luigi
import mc.factories as mcfac
import numpy as np
from mc.sim import SimWrapper, get_diffuse_reflectance
from mc import tissueparser

import commons


sc = commons.ScriptCommons()


class CreateSpectraTask(luigi.Task):
    tissue_file = luigi.Parameter()
    batch_nr = luigi.IntParameter()
    factory = luigi.Parameter()
    experiment_dict = luigi.Parameter()

    def output(self):
        simulated_data_folder=self._get_folder_name()

        return luigi.LocalTarget(os.path.join(sc.get_full_dir("MC_DATA_FOLDER"),
                                              simulated_data_folder,
                                              str(self.batch_nr) + ".csv"))

    def run(self):
        start = time.time()

        # the specific subfolder where the generated spectra shall be saved
        simulated_data_folder=self._get_folder_name()

        # just to be a bit shorter in the code
        ex = self.experiment_dict

        # the wavelengths to be simulated
        wavelengths = np.arange(float(ex["wavelengths_start"]),
                                float(ex["wavelengths_end"]),
                                float(ex["wavelengths_step"])) * 10**-9

        # create folder for mci files if not exists
        mci_folder = os.path.join(sc.get_full_dir("MC_DATA_FOLDER"), simulated_data_folder,
                                  "mci")
        if not os.path.exists(mci_folder):
            os.makedirs(mci_folder)

        # Setup simulation wrapper
        sim_wrapper = SimWrapper()
        sim_wrapper.set_mcml_executable(os.path.join(ex["path_to_mcml"],
                                                     ex["mcml_executable"]))
        sim_wrapper.set_mci_filename(os.path.join(mci_folder, "Bat_" + str(self.batch_nr) + ".mci"))

        # Setup tissue model
        tissue_model = self.factory.create_tissue_model()
        tissue_model.set_mci_filename(sim_wrapper.mci_filename)
        tissue_model.set_nr_photons(int(ex["nr_photons"]))
        tissue_model._mci_wrapper.set_nr_runs(
            int(ex["nr_elements_in_batch"]) * wavelengths.shape[0])
        tissue_model.create_mci_file()

        tissue_instance = tissueparser.read_tissue_config(self.tissue_file)

        # setup array in which data shall be stored
        batch = self.factory.create_batch_to_simulate()
        batch.set_tissue_instance(tissue_instance)

        # create the tissue samples and return them in dataframe df
        df = batch.create_tissue_samples(int(ex["nr_elements_in_batch"]))
        # add reflectance column to dataframe
        for w in wavelengths:
            df["reflectances", w] = np.NAN

        # Generate MCI file which contains list of all simulations in a Batch
        for i in range(df.shape[0]):
            # set the desired element in the dataframe to be simulated
            base_mco_filename = _create_mco_filename_for(simulated_data_folder,
                                                         self.batch_nr,
                                                         i)
            tissue_model.set_base_mco_filename(base_mco_filename)
            tissue_model.set_tissue_instance(df.loc[i, :])
            tissue_model.update_mci_file(wavelengths)

        # Run simulations for computing reflectance from parameters
        sim_wrapper.run_simulation()

        # get information from created mco files
        for i in range(df.shape[0]):
            for wavelength in wavelengths:
                # for simulation get which mco file was created
                simulation_path = os.path.split(sim_wrapper.mcml_executable)[0]
                base_mco_filename = _create_mco_filename_for(simulated_data_folder,
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

    def _get_folder_name(self):
        # return the folder where the generated data shall be saved.
        if self.batch_nr < int(self.experiment_dict["nr_train_batches"]):
            postfix = "train"
        else:
            postfix = "test"

        tissue_file_path, tissue_file_name = os.path.split(self.tissue_file)

        mc_folder_name = self.experiment_dict["experiment_name"] + "_" + \
                         os.path.splitext(tissue_file_name)[0] + "_" + \
                         postfix
        return mc_folder_name


def _create_mco_filename_for(prefix, batch, simulation):
    return str(prefix) + "_Bat_" + str(batch) + "_Sim_" + str(simulation) + "_"


def main(args):
    experiment_dict = commons.read_configuration_dict(args[1])

    # create a folder for the results if necessary
    sc.set_root(experiment_dict["root_path"])
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
    BATCH_NUMBERS = np.arange(0, int(experiment_dict["nr_batches"]), 1)

    for i in BATCH_NUMBERS:
        task = CreateSpectraTask(tissue_file=args[2],
                                 batch_nr=i,
                                 factory=mcfac.GenericMcFactory(),
                                 experiment_dict=experiment_dict)
        w.add(task)
        w.run()


if __name__ == '__main__':
    main(sys.argv)
