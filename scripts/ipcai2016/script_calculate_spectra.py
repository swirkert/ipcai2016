
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
import cProfile

import luigi
import mc.factories as mcfac
import numpy as np
from mc.create_spectrum import create_spectrum
from mc.sim import SimWrapper, get_diffuse_reflectance

import commons


# parameter setting
NR_BATCHES = 15
NR_ELEMENTS_IN_BATCH = 1000
# the wavelengths to be simulated
WAVELENGHTS = np.arange(450, 978, 2) * 10 ** -9
NR_PHOTONS = 10 ** 6

# experiment configuration
#MCI_FILENAME = "./temp.mci"
#MCO_FILENAME = "temp.mco"
# this path definitly needs to be adapted by you
PATH_TO_MCML = "/media/avemuri/DEV/MCML/fast-gpumcml/"
EXEC_MCML = "gpumcml.sm_20"
OUTPUT_ROOT_PATH = "/media/avemuri/DEV/IPCAI2016-Seb/SimulatedData"

sc = commons.ScriptCommons()


class CreateSpectraTask(luigi.Task):
    df_prefix = luigi.Parameter()
    batch_nr = luigi.IntParameter()
    nr_samples = luigi.IntParameter()
    factory = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(os.path.join(sc.get_full_dir("MC_DATA_FOLDER"),
                                              self.df_prefix + "_" +
                                              str(self.batch_nr) + ".csv"))

    def run(self):
        start = time.time()
        # Setup simulation wrapper
        sim_wrapper = SimWrapper()
        sim_wrapper.set_mcml_executable(os.path.join(PATH_TO_MCML, EXEC_MCML))
        sim_wrapper.set_mci_filename(str(self.df_prefix)
                                     + "_Bat_" + str(self.batch_nr) + ".mci")



        # Setup tissue model
        tissue_model = self.factory.create_tissue_model()
        tissue_model.set_mci_filename(sim_wrapper.mci_filename)
        #tissue_model.set_mco_filename(MCO_FILENAME)
        tissue_model.set_nr_photons(NR_PHOTONS)
        tissue_model._mci_wrapper.set_nr_runs(
            NR_ELEMENTS_IN_BATCH * WAVELENGHTS.shape[0])
        tissue_model.create_mci_file()
        # Organize the files
        # os.rename(str(self.df_prefix)
        #           + "_Bat_" + str(self.batch_nr) + ".mci",
        #           )



        # setup array in which data shall be stored
        batch = self.factory.create_batch_to_simulate()
        batch.create_parameters(self.nr_samples)
        # dataframe created by batch:
        df = batch.df
        # add reflectance column to dataframe
        for w in WAVELENGHTS:
            df["reflectances", w] = np.NAN

        # Generate MCI file which contains list of all simulations in a Batch
        for i in range(df.shape[0]):
            # set the desired element in the dataframe to be simulated
            base_mco_filename = \
                str(self.df_prefix) + "_Sim_" + str(i) + "_Bat_" \
                + str(self.batch_nr) + "_"
            tissue_model.set_base_mco_filename(base_mco_filename)
            tissue_model.set_dataframe_row(df.loc[i, :])
            tissue_model.update_mci_file(WAVELENGHTS)


        # Run simulations for computing reflectance from parameters
        sim_wrapper.run_simulation()

        # Load the MCO files and compute
        for i in range(df.shape[0]):
            for wavelength in WAVELENGHTS:
                simulation_path = os.path.split(sim_wrapper.mcml_executable)[0]
                base_mco_filename = \
                    str(self.df_prefix) + "_Sim_" + str(i) + \
                    "_Bat_" + str(self.batch_nr) + "_"
                mco_filename = base_mco_filename + str(wavelength) + '.mco'
                df["reflectances",wavelength][i] = \
                    get_diffuse_reflectance(os.path.join(simulation_path,mco_filename))
                os.remove(os.path.join(simulation_path,mco_filename))


        f = open(self.output().path, 'w')
        df.to_csv(f)

        end = time.time()
        logging.info("time for creating batch of mc data: %.f s" %
                     (end - start))


if __name__ == '__main__':

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
        colon_test_task = CreateSpectraTask("ipcai_mean_scattering_test",
                                       i,
                                       NR_ELEMENTS_IN_BATCH,
                                       mcfac.ColonMuscleMeanScatteringFactory())
        colon_train_task = CreateSpectraTask("ipcai_mean_scattering_train",
                                       i,
                                       NR_ELEMENTS_IN_BATCH,
                                       mcfac.ColonMuscleMeanScatteringFactory())
        # generic_task = CreateSpectraTask("ipcai_generic_mean_scattering_test",
        #                                i,
        #                                NR_ELEMENTS_IN_BATCH,
        #                                mcfac.GenericMeanScatteringFactory())
        w.add(colon_train_task)
        w.add(colon_test_task)
        #w.add(generic_task)
        w.run()

