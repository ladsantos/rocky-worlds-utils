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


__all__ = ["read_hlsp", "fixed_bin_width", "adapt"]


# Reads an HLSP spectrum
def read_hlsp(filename, prefix='.'):
    """
    Read a RWDDT HLSP file.

    Parameters
    ----------
    filename : ``str``
        Filename of RWDDT HLSP file.

    prefix : ``str``, optional
        Prefix of filename. Default is '.'.

    Returns
    -------
    wavelength : ``numpy.ndarray``
        Array containing wavelengths in unit of Angstrom.

    flux : ``numpy.ndarray``
        Array containing fluxes in unit of erg/s/cm^2/Angstrom.

    flux_error : ``numpy.ndarray``
        Array containing flux uncertainties in erg/s/cm^2/Angstrom.
    """
    hlsp_file = os.path.join(prefix, filename)
    data = fits.getdata(hlsp_file)
    wavelength = data["WAVELENGTH"]
    flux = data["FLUX"]
    flux_error = data["FLUXERROR"]
    return wavelength, flux, flux_error


# Spectrum binning
def fixed_bin_width(wavelength, flux, flux_error, bin_width=1.0):
    """
    Bins a spectrum to a given bin width.

    Parameters
    ----------
    wavelength : ``numpy.ndarray``
        Array containing wavelengths in unit of Angstrom.

    flux : ``numpy.ndarray``
        Array containing fluxes in unit of erg/s/cm^2/Angstrom.

    flux_error : ``numpy.ndarray``
        Array containing flux uncertainties in erg/s/cm^2/Angstrom.

    bin_width : ``float``, optional
        Bin with in unit of Angstrom. Default is 1.0.

    Returns
    -------
    binned_wavelength : ``numpy.ndarray``
        Binned wavelength array.

    binned_flux : ``numpy.ndarray``
        Binned flux array.

    binned_error : ``numpy.ndarray``
        Binned flux uncertainty array.
    """
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
    binned_wavelength = np.arange(wavelength_start + bin_width / 2,
                                      wavelength_stop - bin_width / 2,
                                      bin_width)
    resampler = FluxConservingResampler()
    new_spec = resampler(input_spec, binned_wavelength * wave_unit)
    binned_flux = new_spec.flux.value
    binned_error = new_spec.uncertainty

    return binned_wavelength, binned_flux, binned_error


# Adaptive binning
def adapt(wavelength, flux, flux_error, wavelength_threshold=2750., margin=20.):
    """
    The adaptive binning iteratively bins a spectrum within a defined wavelength
    threshold to a given bin width to avoid negative fluxes, and then upsamples
    it to the original wavelength grid.

    Parameters
    ----------
    wavelength : ``numpy.ndarray``
        Array containing wavelengths in unit of Angstrom.

    flux : ``numpy.ndarray``
        Array containing fluxes in unit of erg/s/cm^2/Angstrom.

    flux_error : ``numpy.ndarray``
        Array containing flux uncertainties in erg/s/cm^2/Angstrom.

    wavelength_threshold : ``float``, optional

    margin : ``float``, optional

    Returns
    -------

    """
    delta_wl = wavelength_threshold - min(wavelength)

    denominators = np.array([2500, 1000, 500, 250, 100, 50, 25, 10, 5, 2.5])
    bin_widths = delta_wl / denominators

    # wavelength_threshold += 1E-5  # little trick
    # wavelength_section = wavelength[wavelength < wavelength_threshold + margin]
    # flux_section = flux[wavelength < wavelength_threshold + margin]
    # error_section = flux_error[wavelength < wavelength_threshold + margin]
    #
    # test = np.any(flux_section < 0)
    # if not test:
    #     raise ValueError('Adaptive binning is not necessary for this data.')

    for bin_width in bin_widths:
        wavelength_section = wavelength[wavelength < wavelength_threshold + bin_width / 2]
        flux_section = flux[wavelength < wavelength_threshold + bin_width / 2]
        error_section = flux_error[wavelength < wavelength_threshold + bin_width / 2]
        test = np.any(flux_section < 0)

        if test:
            # Perform the resample to a lower resolution
            new_wavelength, new_flux, new_flux_error = (
                fixed_bin_width(wavelength_section, flux_section, error_section,
                                bin_width=bin_width)
            )
            # Now, upsample if bin_width > 1.0
            if bin_width > 1.0:
                up_wavelength, up_flux, up_flux_error = (
                    fixed_bin_width(new_wavelength, new_flux, new_flux_error,
                                    bin_width=1.0))
                # We have to do a bit of a trick here so that the up_wavelength
                # array includes the whole original wavelength range
                up_wavelength_full = np.arange(min(wavelength) + 0.5,
                                               wavelength_threshold,
                                               1.0)
                up_flux_full = np.interp(up_wavelength_full,
                                         up_wavelength,
                                         up_flux,
                                         left=up_flux[0],
                                         right=up_flux[-1])
                # up_flux_error_full = np.interp(up_wavelength_full,
                #                                up_wavelength,
                #                                up_flux_error,
                #                                left=up_flux_error[0],
                #                                right=up_flux_error[-1])
            else:
                up_wavelength_full, up_flux_full, up_flux_error_full = (
                    new_wavelength, new_flux, new_flux_error)
        else:
            pass

    # Assign the upsampled spectrum section to a copy of the original
    new_wavelength = np.concatenate((up_wavelength_full[up_wavelength_full < wavelength_threshold],
                                     wavelength[wavelength > wavelength_threshold]))
    new_flux = np.concatenate((up_flux_full[up_wavelength_full < wavelength_threshold],
                               flux[wavelength > wavelength_threshold]))
    # new_flux_error = np.concatenate((up_flux_error_full[up_wavelength_full < wavelength_threshold],
    #                                  up_flux_error[wavelength > wavelength_threshold]))

    return new_wavelength, new_flux#, new_flux_error