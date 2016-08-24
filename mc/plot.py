
import pandas as pd


def plot(batch, m_to_nm=False, norm_to_one=False, **kwargs):
    """
    Plot all reflectances of a batch
    Args:
        batch: the batch created by batches.py
        m_to_nm: transform nanometer values to m.
        norm_to_one: normalize spectra to a total sum of one
        kwargs: to be forwarded to plot

    Returns:
        a plot of all reflectances in the batch. wavelengths will be taken 1e9
    """

    reflectances = batch["reflectances"]
    if norm_to_one:
        sums = reflectances.sum(axis=1)
        reflectances = reflectances.div(sums, axis=0)

    if m_to_nm:
        reflectances.columns=reflectances.columns.values.astype(float)*10**9

    reflectances.T.plot(**kwargs)