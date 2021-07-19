#!/usr/bin/env python
# -*- coding: utf-8 -*-
from gfs_icbc.interpolation import verticalInterpolation as vint
import os
import shutil
import calendar
import xarray as xr
import netCDF4 as nc
import cftime
import numpy as np
import pandas as pd
import xesmf
import datetime as dt

# turn off warnings
import warnings
warnings.filterwarnings("ignore")

cₚ = 1004.
cᵥ = 717.
γ = cₚ/cᵥ
g = 9.806
Γ = -g/cₚ

def calc_psl(pₛ, zₛ, Tₛ):
    
    psl = pₛ*(1.0 + (Γ * zₛ)/(Tₛ - Γ*zₛ))**(-γ/(γ-1))
    return psl

def calc_pₛ(psl, zₛ, Tₛ):
    
    pₛ = psl*(1.0 + (Γ * zₛ)/(Tₛ - Γ*zₛ))**(γ/(γ-1))
    return pₛ

def calc_tsl(Tₛ, zₛ):
    tsl = Tₛ - Γ*zₛ
    return tsl

def calc_tₛ(tsl, zₛ):
    Tₛ = tsl + Γ*zₛ
    return Tₛ


def generate_icbc_from_template(
    icbc_template_file,
    icbc_year,
    icbc_month,
    steps_per_day = 4,
    output_file_pattern = None,
    clobber = False,
    zero_data = False,
):
    """Generates a RegCM ICBC file, given a template file from the intended run (need to have run icbc previously for this domain).

        input:
        ------

            icbc_template_file  : the path to an ICBC file generated for the intended RegCM domain

            icbc_year           : the numeric year for which to generate the ICBC file

            icbc_month          : the numeric month (1--12) for which to generate the ICBC file

            steps_per_day       : the number of ICBC steps per day (default is almost always correct)
            
            output_file_pattern : the pattern to use for generating ICBC file names; the input ICBC file's pattern is used as default

            clobber             : flags whether to clobber an existing file (simply returns if the file exists otherwise)

            zero_data           : flags whether to zero-out the ICBC fields

        output:
        -------

            icbc_file_path : the full path to the ICBC file created by this routine


        This routine copies the template file and overwrites the dates. The number of timesteps in the file are adjusted as appropriate.
    """

    if output_file_pattern is None:
        output_file_pattern = f"{os.path.basename(icbc_template_file).split('.')[0]}.{{year:04}}{{month:02}}0100.nc"

    try:
        output_file_name = output_file_pattern.format(year = int(icbc_year), month = int(icbc_month))
    except:
        raise RuntimeError("`output_file_name` must have the print template fields `year` and `month`, and icbc_year and icbc_month must be numbers")

    # simply return if the file already exists and we aren't clobbering
    if not clobber:
        if os.path.exists(output_file_name): return output_file_name

    # create the output file
    shutil.copy(icbc_template_file, output_file_name)

    # determine how many timesteps the file should have
    _, days_in_month = calendar.monthrange(icbc_year, icbc_month)

    # set the range of dates to output
    interval = int(24/steps_per_day)
    first_date = dt.datetime(icbc_year, icbc_month, 1, 0)
    last_date = dt.datetime(icbc_year, icbc_month, days_in_month, 24 - interval)
    dates = pd.date_range(first_date, last_date, freq = f"{interval}H").to_pydatetime()

    # open the template file
    with nc.Dataset(output_file_name, "r+") as fio:

        # set the source as GFS
        fio.global_atm_source = "GFS"

        # update the history
        fio.history = f"{dt.datetime.today()} : Created from {os.path.abspath(icbc_template_file)} by {os.path.basename(__file__)}"

        # get the time dimension units
        time_units = fio.variables['time'].units

        # convert the month's times to the file's time units
        times = cftime.date2num(dates, time_units)

        # write the times (this adjusts the # of timesteps in the file)
        fio.variables['time'][:] = times

        # zero-out the ICBC fields
        if zero_data:
            for var in ["ts", "ps", "u", "v", "t", "qv"]:
                fio.variables[var][:] = np.zeros(np.shape(fio.variables[var]))

    return os.path.abspath(output_file_name)

def generate_regridding_files(
    gfs_xr,
    icbc_filename,
    method = "bilinear",
    overwrite_weights = False
    ):
    """Generates regridding weights for mapping from the GFS grid to the RCM grid.

    input:
    ------

        gfs_xr              : an already-open xarray object corresponding to a GFS forecast file

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
    weightfiles['dot'] = f"wgts_{method}_GFS_to_{domain_name}_dot.nc"
    weightfiles['cross'] = f"wgts_{method}_GFS_to_{domain_name}_cross.nc"

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
            gfs_xr,
            icbc_xr.rename(dict(dlat="lat",dlon="lon")).drop(['mask']),
            method = method,
            filename = weightfiles['dot'],
            reuse_weights = reuse_weights
        )

        # generate the cross grid regridder
        regridders['cross'] = xesmf.Regridder(
            gfs_xr,
            icbc_xr.rename(dict(xlat="lat",xlon="lon")).drop(['mask']),
            method = method,
            filename = weightfiles['cross'],
            reuse_weights = reuse_weights
        )

    return regridders



def open_gfs_file(gfs_file):
    """Opens a local or remote GFS file."""

    if "https://" in gfs_file or "http://" in gfs_file:
        # assume we are working with data from an OpenDAP server
        pressure_xr = xr.Dataset()
        surface_xr = xr.Dataset()
        with xr.open_dataset(gfs_file) as fin:
            # translate the dimension name to a grib-like name
            frename = fin.rename({"isobaric1" : "isobaricInhPa"}).squeeze()

            # translate variables to have grib-like names
            pressure_xr['u'] = frename['u-component_of_wind_isobaric']
            pressure_xr['v'] = frename['v-component_of_wind_isobaric']
            pressure_xr['t'] = frename['Temperature_isobaric']
            pressure_xr['q'] = frename['Specific_humidity_isobaric']
            pressure_xr['isobaricInhPa'] = frename['isobaricInhPa']/100

            surface_xr['sp'] = frename['Pressure_surface']
            surface_xr['t'] = frename['Temperature_surface']
            surface_xr['orog'] = frename['Geopotential_height_surface']
    
        # remove any excess dimensions
        pressure_xr = pressure_xr.squeeze()
        surface_xr = surface_xr.squeeze()
    else:
        # assume we are working with a grib file
        # open the grib file twice: once for the pressure-level group and once for the surface group
        filter_dict = {'typeOfLevel': 'isobaricInhPa'}
        pressure_xr = xr.open_dataset(gfs_file, engine = 'cfgrib', filter_by_keys = filter_dict)
        filter_dict = {'typeOfLevel': 'surface'}
        surface_xr = xr.open_dataset(gfs_file, engine = 'cfgrib', filter_by_keys = filter_dict)
    
    return pressure_xr, surface_xr


def interpolate_gfs_to_regcm(
    gfs_file,
    icbc_file,
    method = 'bilinear',
    overwrite_weights = False,
):
    """Interploates GFS state data for one timestep into a RegCM ICBC file.

        input:
        ------

            gfs_file            : An xarray-compatible path to a GFS forecast grib file

            icbc_file           : The path to an ICBC file.
                                  Note: data in this file WILL be overwritten.
                                  Consider using generate_icbc_from_template() to create a safe version.

            method              : the regridding method (see xesmf.Regridder()) 

            overwrite_weights   : flag whether to overwrite the regridding weights 
                                  (they are reused if they exist otherwise)

        output:
        -------

            Overwrites the matching timestamp in icbc_file with data interpolated from GFS.

            Raises an error if the date from the GFS forecast doesn't exist in the ICBC file.
    """
    # open the GFS file; wraps xarray to allow handling of http URLs
    pressure_xr, surface_xr = open_gfs_file(gfs_file)

    # generate regridding weights
    regridders = generate_regridding_files(surface_xr, icbc_file, method = method, overwrite_weights = overwrite_weights)

    # extract surface fields from GFS
    pₛ = surface_xr["sp"]
    Tₛ = surface_xr["t"]
    zₛ = surface_xr["orog"]
    psl = calc_psl(pₛ, zₛ, Tₛ)

    # interpolate sea-level pressure and temperature to the RCM grid
    psl_rcm = regridders['dot'](calc_psl(surface_xr['sp'], surface_xr['orog'], surface_xr['t']))
    tsl_rcm = regridders['dot'](calc_tsl(surface_xr['t'], surface_xr['orog']))

    # convert sea-level pressure and temperature back to their orographic values
    with xr.open_dataset(icbc_file) as icbc_template_xr:
        zₛ_rcm = icbc_template_xr['topo']
        tₛ_rcm = calc_tₛ(tsl_rcm, zₛ_rcm)
        pₛ_rcm = calc_pₛ(psl_rcm, zₛ_rcm, tₛ_rcm)

        # calculate RCM pressure
        pₜ = icbc_template_xr['ptop'].values
        σ = icbc_template_xr['kz'].values[:,np.newaxis, np.newaxis]
        p_rcm = (pₛ_rcm.values[np.newaxis,:,:] - pₜ)*σ + pₜ


    # set the pressure levels from which to interpolate
    plevs = pressure_xr['isobaricInhPa'].values * 100

    # map the GFS variable names to RegCM variable names
    gfs2rcm_var_mapping = dict(t = 't', q = 'qv', u = 'u', v = 'v')

    # open the ICBC file for overwriting
    with nc.Dataset(icbc_file, "r+") as fio:

        # determine the date of the GFS file
        gfs_date = pd.Timestamp(pressure_xr['time'].values).to_pydatetime()

        # convert this to the file's units
        gfs_timeval = cftime.date2num(gfs_date, fio.variables['time'].units)

        # determine the time index of the file
        itime = cftime.time2index(gfs_timeval, fio.variables['time'])

        # sanity check on time
        if itime < 0 or itime >= len(fio.dimensions['time']):
            raise RuntimeError(f"Date {gfs_date} is out of bounds for {icbc_file}")


        # loop over variables
        for var in list(gfs2rcm_var_mapping):
            # set the regridding grid
            if var in ['u', 'v']:
                grid = "cross"
            else:
                grid = "dot"

            # regrid the gfs field horizontally to the RCM grid
            var_plev = regridders[grid](pressure_xr[var])

            # regrid the field vertically to the RCM grid
            var_sigma = vint.interpolatePressureLevels(var_plev.values, plevs, p_rcm, doExtrapolation=True, interpolationDimension=0)

            # get the regcm variable name
            rcm_var = gfs2rcm_var_mapping[var]

            # insert the variable into the ICBC file at the appropriate time index
            fio.variables[rcm_var][itime,...] = var_sigma

        # also put the surface pressure and temperature into the file 
        fio.variables['ps'][itime,...] = pₛ_rcm.values / 100 # also convert to hPa
        fio.variables['ts'][itime,...] = tₛ_rcm.values


def generate_gfs_icbc(
    icbc_template_file,
    start_date,
    end_date,
    steps_per_day = 4,
    gfs_path_template = "https://www.ncei.noaa.gov/thredds/dodsC/model-gfs-g4-anl-files/{year:04}{month:02}/{year:04}{month:02}{day:02}/gfs_4_{year:04}{month:02}{day:02}_{hour:04}_000.grb2",
    method = 'bilinear',
    overwrite_weights = False,
):
    """Generates ICBC files for RegCM from the GFS forecast.

        input:
        ------

            icbc_template_file  : the path to an ICBC file generated for the intended RegCM domain
            
            start_date          : a datetime object giving the initial date for which to create ICBCs

            end_date            : a datetime object giving the end date (exclusive) for which to create ICBCs

            steps_per_day       : the number of ICBC steps per day (default is almost always correct)

            gfs_path_template   : a template for a file path or URL that will direct to a GFS file
                                  The template needs to be able to use string.format() with the fields 
                                  year, month, and day.

            method              : the regridding method (see xesmf.Regridder()) 

            overwrite_weights   : flag whether to overwrite the regridding weights 
                                  (they are reused if they exist otherwise)

        output:
        -------

            icbc_file_paths : the absolute paths to the ICBC files produced by this routine

    """

    interval = int(24/steps_per_day)
    # set the range of dates for which to run the ICBC creation (excluding the last date)
    dates = pd.date_range(start_date, end_date, freq = f"{interval}H").to_pydatetime()

    icbc_files = []
    # loop over dates
    for date in dates:
        print(date)
        # set the GFS file for this date
        gfs_file = gfs_path_template.format(year = date.year, month = date.month, day = date.day, hour = date.hour*100)

        # set/create the ICBC file for this month (don't clobber so that we can run this for each date in the month)
        icbc_file = generate_icbc_from_template(icbc_template_file,date.year, date.month, steps_per_day, clobber = False, zero_data = True)
        icbc_files.append(icbc_file)

        # regrids into that ICBC file
        interpolate_gfs_to_regcm(gfs_file, icbc_file)
        
    # return the list of ICBC files
    return list(sorted(set(icbc_files)))

if __name__ == "__main__":
    import datetime as dt

    icbc_template_file = "fog_ctd_control_ICBC.2000040100.nc"
    start_date = dt.datetime(2021,6,27)
    end_date = dt.datetime(2021,6,28)

    generate_gfs_icbc(icbc_template_file, start_date, end_date)
