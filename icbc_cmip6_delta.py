#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from gfs_icbc.interpolation import verticalInterpolation as vint
except:
    from interpolation import verticalInterpolation as vint
import os
import shutil
import xarray as xr
import netCDF4 as nc
import numpy as np
import pandas as pd
import xesmf
import datetime as dt
import progress.bar

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

cₚ = 1004.
cᵥ = 717.
γ = cₚ/cᵥ
g = 9.806
Γ = -g/cₚ

def generate_delta_icbc_from_template(
    icbc_template_file,
    icbc_year,
    icbc_month,
    output_file_pattern = None,
    clobber = False,
):
    """Generates a RegCM ICBC file, given a template file from the intended run (need to have run icbc previously for this domain).

        input:
        ------

            icbc_template_file  : the path to an ICBC file generated for the intended RegCM domain

            steps_per_day       : the number of ICBC steps per day (default is almost always correct)
            
            output_file_pattern : the pattern to use for generating ICBC file names; a modified version of the input ICBC file's pattern is used as default

            clobber             : flags whether to clobber an existing file (simply returns if the file exists otherwise)

        output:
        -------

            icbc_file_path : the full path to the ICBC file created by this routine


        This routine copies the template file and overwrites the dates.
    """

    if output_file_pattern is None:
        output_file_pattern = f"pgw_{os.path.basename(icbc_template_file)}"

    try:
        output_file_name = output_file_pattern.format(year = int(icbc_year), month = int(icbc_month))
    except:
        raise RuntimeError("`output_file_name` must have the print template fields `year` and `month`, and icbc_year and icbc_month must be numbers")

    # simply return if the file already exists and we aren't clobbering
    if not clobber:
        if os.path.exists(output_file_name): return output_file_name

    # create the output file
    shutil.copy(icbc_template_file, output_file_name)

    return os.path.abspath(output_file_name)

def generate_regridding_files(
    cmip6_xr,
    icbc_filename,
    method = "bilinear",
    overwrite_weights = False
    ):
    """Generates regridding weights for mapping from the delta grid to the RCM grid.

    input:
    ------

        cmip6_xr            : an already-open xarray object corresponding to a CMIP forecast file

        icbc_filename       : an ICBC file for the domain to which to regrid

        method              : the regridding method (see xesmf.Regridder()) 

        overwrite_weights   : flag whether to overwrite the regridding weights 
                              (they are reused if they exist otherwise)
    output:
    -------

        regridders : a dictionary of xesmf.Regridder objects: 'dot' corresponds
        to regridding to the dot grid, and 'cross' corresponds to regridding to
        the cross grid
        
    """

    # determine the regcm domain name
    domain_name = os.path.basename(icbc_filename).split("_ICBC.")[0]

    # set the weight filenames
    weightfiles = {}
    weightfiles['dot'] = f"wgts_{method}_CMIP6_to_{domain_name}_dot.nc"
    weightfiles['cross'] = f"wgts_{method}_CMIP6_to_{domain_name}_cross.nc"

    reuse_weights = False
    if not overwrite_weights:
        # if all the weight files exist, then reuse them
        if all([os.path.exists(weightfiles[g]) for g in weightfiles]):
            reuse_weights = True

    # initialize the dictionary of regridding files
    regridders = {}

    # open the ICBC template file
    with xr.open_dataset(icbc_filename) as icbc_xr:
        # generate the dot grid regridder
        regridders['dot'] = xesmf.Regridder(
            cmip6_xr,
            icbc_xr.rename(dict(dlat="lat",dlon="lon")).drop(['mask']),
            method = method,
            filename = weightfiles['dot'],
            reuse_weights = reuse_weights
        )

        # generate the cross grid regridder
        regridders['cross'] = xesmf.Regridder(
            cmip6_xr,
            icbc_xr.rename(dict(xlat="lat",xlon="lon")).drop(['mask']),
            method = method,
            filename = weightfiles['cross'],
            reuse_weights = reuse_weights
        )

    return regridders


def interpolate_cmip6_to_regcm(
    cmip6_file_pattern,
    icbc_file,
    method = 'bilinear',
    delta_sign = -1,
    overwrite_weights = False,
):
    """Interploates CMIP state data for one timestep into a RegCM ICBC file.

        input:
        ------

            cmip6_file_pattern  : a globbable path to netCDF files containing global CMIP6 delta values

            icbc_file           : The path to an ICBC file.
                                  Note: data in this file WILL be overwritten.
                                  Consider using generate_icbc_from_template() to create a safe version.

            method              : the regridding method (see xesmf.Regridder()) 

            delta_sign          : the sign of the delta (multiplied by the delta before adding to ICBC)

            overwrite_weights   : flag whether to overwrite the regridding weights 
                                  (they are reused if they exist otherwise)

        output:
        -------

            Overwrites the matching timestamp in icbc_file with data interpolated from CMIP.

            Raises an error if the date from the CMIP forecast doesn't exist in the ICBC file.
    """
    # open the CMIP file; wraps xarray to allow handling of http URLs
    cmip6_xr = xr.open_mfdataset(cmip6_file_pattern)

    # generate regridding weights
    regridders = generate_regridding_files(cmip6_xr, icbc_file, method = method, overwrite_weights = overwrite_weights)

    # extract surface fields from CMIP
    dTₛ = cmip6_xr["ts"]

    # set the pressure levels from which to interpolate
    # and convert to hPa
    plevs = cmip6_xr['plev'].values / 100

    # convert sea-level pressure and temperature back to their orographic values
    with xr.open_dataset(icbc_file) as icbc_template_xr:

        # map the CMIP variable names to RegCM variable names
        cmip62rcm_var_mapping = dict(ta = 't', hus = 'qv')

        # open the ICBC file for overwriting
        with nc.Dataset(icbc_file, "r+") as fio:

                
            ntime = len(fio.dimensions['time'])
            pbar = progress.bar.IncrementalBar(os.path.basename(icbc_file), max = ntime)

            # loop over the time dimension
            for t in range(ntime):

                # read surface temperature
                tₛ_rcm = icbc_template_xr['ts'][t,...].values
                # skip this timestep if all of the surface temp values are 0
                if np.sum(tₛ_rcm) == 0.0:
                    pbar.next()
                    continue

                # calculate RCM 3D pressure
                pₛ_rcm = icbc_template_xr['ps'][t,...].values
                pₜ = icbc_template_xr['ptop'].values
                σ = icbc_template_xr['kz'].values[:,np.newaxis, np.newaxis]
                p_rcm = ((pₛ_rcm[np.newaxis,:,:] - pₜ)*σ + pₜ)

                # loop over variables
                for var in list(cmip62rcm_var_mapping):
                    # set the regridding grid
                    if var in ['u', 'v']:
                        grid = "cross"
                    else:
                        grid = "dot"

                    # get the regcm variable name
                    rcm_var = cmip62rcm_var_mapping[var]

                    # regrid the cmip6 field horizontally to the RCM grid;
                    dvar_plev = regridders[grid](cmip6_xr[var])
                    
                    # check if we need to interpolate for this timestep
                    rcm_vals = fio.variables[rcm_var][t,...]

                    # get a version of the deltas with fill values added
                    dvar_plev.load()
                    fill_value = 1e36
                    dvar_plev_ma = dvar_plev.to_masked_array().filled(fill_value)

                    # regrid the field vertically to the RCM grid
                    dvar_sigma = vint.interpolatePressureLevels(dvar_plev_ma, plevs, p_rcm, doExtrapolation=True, interpolationDimension=0, fillValue=fill_value)

                    # insert the variable into the ICBC file at the appropriate time index
                    fio.variables[rcm_var][t,...] = rcm_vals + delta_sign*dvar_sigma

                # also put the surface pressure and temperature into the file 
                dTₛ_regrid = regridders['dot'](dTₛ)
                fio.variables['ts'][t,...] =  tₛ_rcm + delta_sign*dTₛ_regrid.values

                pbar.next()

            pbar.finish()



def generate_delta_icbc(
    regcm_domain,
    start_date,
    end_date,
    icbc_file_template = "./input/{domain}_ICBC.{year:04}{month:02}0100.nc",
    cmip6_path_template = "cmip6_deltas/*_Amon_CESM2_historical-piControl_r1i1p1f1_gn_month_{month:02}_delta.nc",
    method = 'bilinear',
    overwrite_weights = False,
    delta_sign = -1,
    clobber = True,
):
    """Generates ICBC files for RegCM from the CMIP forecast.

        input:
        ------

            regcm_domain        : the RegCM domain name

            icbc_file_template  : a string template for generating valid ICBC file paths (these files will be perturbed)

            start_date          : the start date for ICBC files (a pandas.date_range compatible date)

            end_date            : the start date for ICBC files (a pandas.date_range compatible date)
            
            cmip6_path_template  : a template for a file path or URL that will direct to a set of CMIP6 delta fields
                                  The template needs to be able to use string.format() with the field month

            method              : the regridding method (see xesmf.Regridder()) 

            overwrite_weights   : flag whether to overwrite the regridding weights 
                                  (they are reused if they exist otherwise)

            delta_sign          : the sign of the delta (multiplied by the delta before adding to ICBC)

            clobber             : flag whether to clobber existing output files

        output:
        -------

            icbc_file_paths : the absolute paths to the ICBC files produced by this routine

    """

    # set the range of dates for which to run the ICBC creation (excluding the last date)
    dates = pd.date_range(start_date, end_date, freq = "1M").to_pydatetime()

    icbc_files = []
    # loop over dates
    for date in dates:
        # set the CMIP file for this date
        cmip6_file_pattern = cmip6_path_template.format(month = date.month)

        # set/create the ICBC file for this month
        icbc_input_file = icbc_file_template.format(domain = regcm_domain, year = date.year, month = date.month)
        icbc_file = generate_delta_icbc_from_template(icbc_input_file,date.year, date.month, clobber = clobber)
        icbc_files.append(icbc_file)

        # regrid into that ICBC file
        interpolate_cmip6_to_regcm(cmip6_file_pattern, icbc_file, method = method, delta_sign = delta_sign, overwrite_weights = overwrite_weights)
        
    # return the list of ICBC files
    return list(sorted(set(icbc_files)))

if __name__ == "__main__":

    regcm_domain = "wus_2021_ctrl_even_more_north"
    start_date = "2021-06"
    end_date = "2021-08"

    generate_delta_icbc(regcm_domain, start_date, end_date)
