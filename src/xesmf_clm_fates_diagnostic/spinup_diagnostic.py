import numpy as np

import xarray as xr

import matplotlib.pyplot as plt

import os, sys, glob

# import netCDF4 as nc4
import warnings

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs

from .plotting_methods import make_se_regridder, regrid_se_data, make_bias_plot
from .infrastructure_help_functions import setup_nested_folder_structure_from_dict, read_pam_file#, clean_empty_folders_in_tree
from  .misc_help_functions import get_unit_conversion_and_new_label

DEFAULT_EQUILIBRIUM_DICTIONARY  = {
    "TOTECOSYSC" : 0.02,
    "TOTSOMC": 0.02,
    "TOTVEGC": 0.02,
    "TLAI": 0.02,
    "GPP": 0.02, 
    "TWS": 0.001,
    "H2OSNO": 1.0,
    "thresh_area": 3.0,
    "totecosysc_thresh": 1.0
}

class SpinupTestObject:
    
    def __init__(
    self, datapath, equilibration_dictionary = None
    ):
        self.datapath = datapath
        if equilibration_dictionary is None:
            self.equilibrium_dictionary = DEFAULT_EQUILIBRIUM_DICTIONARY.copy()
        self.filelist = self.get_clm_h0_filelist(self)

    def get_clm_h0_filelist(self):
        """
        Get a list of all the files for the case

        Returns
        -------
        list
            with paths to all the clm2.h0 files
        """
        #(f"{self.datapath}*.clm2.h0.*.nc")
        return glob.glob(f"{self.datapath}*.clm2.h0.*.nc")

    def make_data_for_tests(self):
        pass
        


    
