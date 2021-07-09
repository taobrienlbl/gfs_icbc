import xarray as xr
import cartopy

def get_regcm_projection(input_file_name):
    """
        input:
        ------
        
        input_file_name : a path to a regcm file
        
        
        output:
        -------
        
        rcm_crs : a cartopy projection
        
        This assumes that the 'crs' variable is in the file.
        
        This assumes that we are using a lambert conformal conic projection.
        
    """
    
    # open input file
    input_xr = xr.open_dataset(input_file_name)
    
    # Set up the map projection based on the 'crs' variable in the file
    # this assumes that 'crs' in the file
    rcrs = input_xr['crs'].attrs
    globe = cartopy.crs.Globe(ellipse='sphere',
                              semimajor_axis = rcrs['semi_major_axis'],
                              semiminor_axis = rcrs['semi_major_axis'])
    rcm_crs = cartopy.crs.LambertConformal(central_longitude = rcrs['longitude_of_central_meridian'],
                                           central_latitude  = rcrs['latitude_of_projection_origin'],
                                           standard_parallels= rcrs['standard_parallel'],
                                           false_easting     = rcrs['false_easting'],
                                           false_northing    = rcrs['false_northing'],
                                           globe = globe
                                          )
    
    # return the projection information
    return rcm_crs