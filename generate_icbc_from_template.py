#!/usr/bin/env python
import netCDF4 as nc
import shutil
import datetime as dt
import calendar
import cftime
import pandas as pd
import os

def generate_icbc_from_template(
    icbc_template_file,
    icbc_year,
    icbc_month,
    steps_per_day = 4,
    output_file_pattern = None
):
    """Generates a RegCM ICBC file, given a template file from the intended run (need to have run icbc previously for this domain).

        input:
        ------

            icbc_template_file  : the path to an ICBC file generated for the intended RegCM domain

            icbc_year           : the numeric year for which to generate the ICBC file

            icbc_month          : the numeric month (1--12) for which to generate the ICBC file

            steps_per_day       : the number of ICBC steps per day (default is almost always correct)
            
            output_file_pattern : the pattern to use for generating ICBC file names; the input ICBC file's pattern is used as default

        output:
        -------

            icbc_file_path : the full path to the ICBC file created by this routine


        This routine copies the template file and overwrites the dates. The number of timesteps in the file are adjusted as appropriate.
    """

    if output_file_pattern is None:
        output_file_pattern = f"{os.path.basename(icbc_template_file).split('.')[0]}{{year:04}}{{month:02}}0100.nc"

    try:
        output_file_name = output_file_pattern.format(year = int(icbc_year), month = int(icbc_month))
    except:
        raise RuntimeError("`output_file_name` must have the print template fields `year` and `month`, and icbc_year and icbc_month must be numbers")

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

    return os.path.abspath(output_file_name)


if __name__ == "__main__":

    import sys

    template_file = sys.argv[1]
    icbc_year = int(sys.argv[2])
    icbc_month = int(sys.argv[3])

    print(generate_icbc_from_template(template_file, icbc_year, icbc_month))

