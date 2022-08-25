from netCDF4 import Dataset

def print_nc_dict(
    nc  # netcdf dataset
):
    """Print each variable's dict of a given nc file

    :param nc: Dataset (nc file) from which values are extracted
    :type nc: Dataset
    """
    # For each variable, print its dict
    for i in nc.variables:
        print("========= VARIABLE: ", i," =========")
        print("shape: ", nc.variables[i].shape)
        ordered_dict = nc.variables[i].__dict__
        for key, value in ordered_dict.items():
            pretty_key = key[:] + " " * (20 - len(key))
            print("Key: {0}|  Value: {1}".format(pretty_key,value))
