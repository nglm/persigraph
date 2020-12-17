import unittest
from DataAnalysis.statistics import *
from netCDF4 import Dataset
import numpy as np
import sys
from contextlib import contextmanager
from io import StringIO


nc = Dataset('/home/natacha/Documents/Work/Data/Bergen/ec.ens.2020011400.sfc.meteogram.nc','r')
d = 6

# Input
list_ind_members = [None,
                    np.arange(1),
                    np.array([0,2,3]),
                    np.arange(50) ]

# Input
list_ind_long = [None,
                 np.arange(1),
                 np.array([1,3,4]),
                 np.arange(21)]

# Input
list_ind_lat = [None,
                np.arange(1),
                np.array([0,3]),
                np.arange(21)]

# Input
list_var_name = [None,
                 ["msl"],
                 ["t2m","msl"],
                 ["t2m","d2m","msl","u10","v10","tcwv"],
                 ]

# Combined with list_ind_members
list_shape_exp = [[(51, 50, 21, 21)]*d,
                  [(51, 1, 21, 21)]*d,
                  [(51, 3, 21, 21)]*d,
                  [(51, 50, 21, 21)]*d,]

# Combined with list_var_name
list_list_var = [
    extract_variables(
        nc,
        var_names=names,
    )[0] for names in list_var_name]

# Combined with list_var_name
# Combined with list_ind_members
# Combined with list_ind_long
# Combined with list_ind_lat
complex_list_list_var = [
    extract_variables(
        nc,
        var_names=list_var_name[i],
        ind_time=None,
        ind_members=list_ind_members[i],
        ind_long=list_ind_long[i],
        ind_lat=list_ind_lat[i],
        descr=False
    )[0] for i in range(len(list_var_name))]


# Combined with list_var_name
# Combined with list_list_var
list_tuple_scalers_stand_var = [
    standardize(
        list_var = list_var,
        each_loc = False,
    )
    for list_var in list_list_var]
(list_list_scalers, list_list_stand_var) = (
    [list_scalers for (list_scalers, _) in list_tuple_scalers_stand_var],
    [list_stand_var for (_, list_stand_var) in list_tuple_scalers_stand_var])

# Combined with complex_list_var
# Combined with list_var_name
# Combined with list_ind_members
# Combined with list_ind_long
# Combined with list_ind_lat
complex_list_tuple_scalers_stand_var = [
    standardize(
        list_var = complex_list_var,
        each_loc = False,
    )
    for complex_list_var in complex_list_list_var]
(complex_list_list_scalers, complex_list_list_stand_var) = (
    [list_scalers for (list_scalers, _) in complex_list_tuple_scalers_stand_var],
    [list_stand_var for (_, list_stand_var) in complex_list_tuple_scalers_stand_var])

# Combined with list_var_name
list_var_distrib_exp = [np.array([
    np.array(nc.variables[name_i]).flatten() for name_i in list_var_name[i]])
                        for i in range(1,len(list_var_name)) ]
list_var_distrib_exp.insert(0, list_var_distrib_exp[-1])

list_mean_var_all_loc = [np.mean(var.flatten()) for var in list_list_var[-1]]
list_std_var_all_loc = [np.std(var.flatten()) for var in list_list_var[-1]]



class TestExtract_variables(unittest.TestCase):
    def test_extract_variables_None(self):
        var_names_exp = ["t2m","d2m","msl","u10","v10","tcwv"]
        (list_var, var_names) = extract_variables(
            nc,
            var_names=None,
            ind_time=None,
            ind_members=None,
            ind_long=None,
            ind_lat=None,
            descr=False)
        list_shape = [var.shape for var in list_var]
        self.assertEqual(len(list_var), d)
        self.assertEqual(len(var_names), d)
        self.assertEqual(list_shape, list_shape_exp[0])
        self.assertEqual(var_names, var_names_exp)

    def test_extract_variables_one_member(self):
        var_names_exp = ["t2m","d2m","msl","u10","v10","tcwv"]
        (list_var, var_names) = extract_variables(
            nc,
            var_names=None,
            ind_members=list_ind_members[1])
        list_shape = [var.shape for var in list_var]
        self.assertEqual(len(list_var), d)
        self.assertEqual(len(var_names), d)
        self.assertEqual(list_shape, list_shape_exp[1])
        self.assertEqual(var_names, var_names_exp)

    def test_extract_variables_some_members(self):
        var_names_exp = ["t2m","d2m","msl","u10","v10","tcwv"]
        var_exp = np.array(nc.variables["msl"])[:,3,:,:]
        with open("test_output_extract_variables_01.txt", "r") as file:
            output_expected = file.read()
        with captured_output() as (out, err):
            (list_var, var_names) = extract_variables(
                nc,
                var_names=["t2m","d2m","msl","u10","v10","tcwv"],
                ind_members=list_ind_members[2],
                descr=True)
        # This can go inside or outside the `with` block
        output = out.getvalue().strip()

        list_shape = [var.shape for var in list_var]
        self.assertEqual(len(list_var), d)
        self.assertEqual(len(var_names), d)
        self.assertEqual(list_shape, list_shape_exp[2])
        self.assertEqual(var_names, var_names_exp)
        self.assertTrue((list_var[2][:,-1,:,:] == var_exp).all())
        self.assertEqual(output, output_expected)

    def test_extract_variables_all_members(self):
        var_names_exp = ["t2m","d2m","msl","u10","v10","tcwv"]
        (list_var, var_names) = extract_variables(
            nc,
            var_names=None,
            ind_members=list_ind_members[3])
        list_shape = [var.shape for var in list_var]
        self.assertEqual(len(list_var), d)
        self.assertEqual(len(var_names), d)
        self.assertEqual(list_shape, list_shape_exp[3])
        self.assertEqual(var_names, var_names_exp)


class TestExtract_var_distrib(unittest.TestCase):

    def test_extract_var_distrib_None(self):
        var_distrib = extract_var_distrib(
            list_var = list_list_var[0],
            descr = False,
            )
        self.assertTrue((var_distrib == list_var_distrib_exp[0]).all())

    def test_extract_var_distrib_one_var(self):
        var_distrib = extract_var_distrib(
            list_var = list_list_var[1],
            descr = False,
            )
        self.assertTrue((var_distrib == list_var_distrib_exp[1]).all())

    def test_extract_var_distrib_some_var(self):
        var_distrib = extract_var_distrib(
            list_var = list_list_var[2],
            descr = False,
            )
        self.assertTrue((var_distrib == list_var_distrib_exp[2]).all())

    def test_extract_var_distrib_all_var(self):
        var_distrib = extract_var_distrib(
            list_var = list_list_var[3],
            descr = False,
            )
        self.assertTrue((var_distrib == list_var_distrib_exp[3]).all())


class TestStandardize(unittest.TestCase):

    def test_standardize_all_var(self):
        (list_scalers, list_stand_var) = standardize(
            list_var=list_list_var[-1]
        )
        d = len(list_list_var[-1])
        for i in range(d):
            self.assertEqual(list_scalers[i].mean_[0], list_mean_var_all_loc[i])
            self.assertEqual(list_scalers[i].scale_[0], list_std_var_all_loc[i])

class TestGet_list_spread(unittest.TestCase):

    def test_get_list_spread_one_var_all_loc(self):
        d = len(list_list_stand_var[0])
        list_shape_spread_exp = [(51,21,21)]*d
        list_var = [np.swapaxes(var, 0,1) for var in list_list_stand_var[0]]
        list_spread = get_list_spread(
            list_var = list_var,
            arg_spread = False
            )
        for i in range(d):
            self.assertEqual(list_spread[i][0].shape, list_shape_spread_exp[i])
            self.assertEqual(list_spread[i][1].shape, list_shape_spread_exp[i])

    def test_get_list_spread_some_var_all_loc(self):
        d = len(list_list_stand_var[1])
        list_shape_spread_exp = [(51,21,21)]*d
        list_var = [np.swapaxes(var, 0,1) for var in list_list_stand_var[1]]
        list_spread = get_list_spread(
            list_var = list_var,
            arg_spread = False
            )
        for i in range(d):
            self.assertEqual(list_spread[i][0].shape, list_shape_spread_exp[i])
            self.assertEqual(list_spread[i][1].shape, list_shape_spread_exp[i])

    def test_get_list_spread_all_var_all_loc(self):
        d = len(list_list_stand_var[2])
        list_shape_spread_exp = [(51,21,21)]*d
        list_var = [np.swapaxes(var, 0,1) for var in list_list_stand_var[2]]
        list_spread = get_list_spread(
            list_var = list_var,
            arg_spread = False
            )
        for i in range(d):
            self.assertEqual(list_spread[i][0].shape, list_shape_spread_exp[i])
            self.assertEqual(list_spread[i][1].shape, list_shape_spread_exp[i])

    def test_get_list_spread_one_var_one_loc(self):
        d = len(complex_list_list_stand_var[1])
        list_shape_spread_exp = [(51,)]*d
        list_var = [np.swapaxes(var, 0,1) for var in complex_list_list_stand_var[1]]
        list_spread = get_list_spread(
            list_var = list_var,
            arg_spread = False
            )
        for i in range(d):
            self.assertEqual(list_spread[i][0].shape, list_shape_spread_exp[i])
            self.assertEqual(list_spread[i][1].shape, list_shape_spread_exp[i])

    def test_get_list_spread_some_var_members_loc(self):
        d = len(complex_list_list_stand_var[2])
        list_shape_spread_exp = [(51,3,2)]*d
        list_var = [np.swapaxes(var, 0,1) for var in complex_list_list_stand_var[2]]
        list_spread = get_list_spread(
            list_var = list_var,
            arg_spread = False
            )
        for i in range(d):
            self.assertEqual(list_spread[i][0].shape, list_shape_spread_exp[i])
            self.assertEqual(list_spread[i][1].shape, list_shape_spread_exp[i])

if __name__ == '__main__':
    unittest.main()



