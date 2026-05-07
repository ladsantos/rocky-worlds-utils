#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module contains useful tools to bin HST spectra.

Authors
-------
- Leonardo dos Santos <<ldsantos@stsci.edu>>
"""

from __future__ import division, print_function, absolute_import, unicode_literals

from astropy.io import fits
from astropy.nddata import StdDevUncertainty, InverseVariance
from astropy import units as u
import numpy as np
import os
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler


__all__ = []


def read_hlsp(filename, prefix='.'):
    """
    Read a RWDDT HLSP file.

    Parameters
    ----------
    filename

    prefix

    Returns
    -------

    """
    hlsp_file = os.path.join(prefix, filename)
    data = fits.getdata(hlsp_file)
    wavelength = data["WAVELENGTH"]
    flux = data["FLUX"]
    flux_error = data["FLUXERROR"]
    return wavelength, flux, flux_error


def fixed_bin_width(wavelength, flux, flux_error,
                    bin_width=1.0):
    """

    Parameters
    ----------
    wavelength
    flux
    flux_error
    bin_width

    Returns
    -------

    """
    # original_wavelength, original_flux, original_flux_error = read_hlsp(
    #     filename, prefix)

    flux_unit = u.erg / (u.s * u.cm ** 2 * u.AA)
    wave_unit = u.AA
    if isinstance(flux_error, np.ndarray):
        uncertainty = StdDevUncertainty(flux_error)
    elif isinstance(flux_error, StdDevUncertainty):
        uncertainty = flux_error
    elif isinstance(flux_error, InverseVariance):
        uncertainty = flux_error
    else:
        raise ValueError('flux_error must be either an array or an Astropy '
                         'uncertainty object.')
    input_spec = Spectrum(spectral_axis=wavelength * wave_unit,
                          flux=flux * flux_unit,
                          uncertainty=uncertainty)

    # Bin the spectrum using specutils
    if int(round(min(wavelength))) == int(min(wavelength)):
        wavelength_start = int(min(wavelength))
    else:
        wavelength_start = int(min(wavelength)) + 1
    wavelength_stop = int(max(wavelength))
    binned_wavelength = np.arange(wavelength_start, wavelength_stop, bin_width)
    resampler = FluxConservingResampler()
    new_spec = resampler(input_spec, binned_wavelength * wave_unit)
    binned_flux = new_spec.flux.value
    binned_error = new_spec.uncertainty

    return binned_wavelength, binned_flux, binned_error


def adapt(wavelength, flux, flux_error, threshold=2750, margin=20):
    """

    Parameters
    ----------
    wavelength
    flux
    flux_error
    threshold

    Returns
    -------

    """
    threshold += 1E-5  # little trick
    wavelength_section = wavelength[wavelength < threshold + margin]
    flux_section = flux[wavelength < threshold + margin]
    error_section = flux_error[wavelength < threshold + margin]

    # Perform the resample to a lower resolution
    new_wavelength, new_flux, new_flux_error = (
        fixed_bin_width(wavelength_section, flux_section, error_section,
                        bin_width=20.0)
    )

    # Now, upsample
    up_wavelength, up_flux, up_flux_error = fixed_bin_width(new_wavelength, new_flux, new_flux_error, bin_width=1.0)

    # Assign the upsampled spectrum section to a copy of the original
    new_wavelength = np.concatenate((up_wavelength[up_wavelength < threshold], wavelength[wavelength > threshold]))
    new_flux = np.concatenate((up_flux[up_wavelength < threshold], flux[wavelength > threshold]))
    # new_flux_error = np.concatenate((up_flux_error, flux_error[wavelength > threshold]))

    return new_wavelength, new_flux