import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xarray as xr

#import matplotlib.pyplot as plt

import os, sys, glob
#import netCDF4 as nc4
import shutil
import warnings
warnings.filterwarnings('ignore')

import xesmf as xe
#import cartopy.crs as ccrs

from .plotting_methods import make_se_regridder, regrid_se_data

VAR_LIST_DEFAULT = ["TSA","RAIN","SNOW","TSOI","FPSN","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","BTRAN2","QINFL","QOVER","QRGWL","QDRAI","QINTR","QSOIL","QVEGT","SOILLIQ","SOILICE","SOILPSI","SNOWLIQ","SNOWICE","WA","ZWT","QCHARGE","FCOV","PCO2","NEE","landfrac","area","FSR","PBOT","SNOWDP","FSDS","FSA","FLDS","FIRE","FIRA","FSH","FCTR","FCEV","FGEV","FGR","WT"]

VAR_LIST_CLIM = ["TSA","RAIN","SNOW","FSR","FSDS","FSA","FIRA","FCTR","FCEV","FGEV","QOVER","QDRAI","QRGWL","SNOWDP","FPSN","FSH","FSH_V","FSH_G","TV","TG","TSNOW","SABV","SABG","FIRE","FGR","FSM","TAUX","TAUY","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","BTRAN2","H2OSNO","H2OCAN","SNOWLIQ","SNOWICE","QINFL","QINTR","QDRIP","QSNOMELT","QSOIL","QVEGE","QVEGT","ERRSOI","ERRSEB","FSNO","ERRSOL","ERRH2O","TBOT","TLAKE","WIND","THBOT","QBOT","ZBOT","FLDS","FSDSNDLN","FSDSNI","FSDSVD","FSDSVDLN","FSDSVI","FSDSND","FSRND","FSRNDLN","FSRNI","FSRVD","FSRVDLN","FSRVI","Q2M","TREFMNAV","TREFMXAV","SOILLIQ","SOILICE","H2OSOI","TSOI","WA","WT","ZWT","QCHARGE","FCOV","PCO2","NEE","GPP","NPP","AR","HR","NEP","ER","SUPPLEMENT_TO_SMINN","SMINN_LEACHED","COL_FIRE_CLOSS","COL_FIRE_NLOSS","PFT_FIRE_CLOSS","PFT_FIRE_NLOSS","FIRESEASONL","FIRE_PROB","ANN_FAREA_BURNED","MEAN_FIRE_PROB","PBOT","SNOBCMCL","SNOBCMSL","SNODSTMCL","SNODSTMSL","SNOOCMCL","SNOOCMSL","BCDEP","DSTDEP","OCDEP"]

class XesmfCLMFatesDiagnostics:

    def __init__(self, datapath, weightfile, varlist=None):
        self.datapath = datapath
        self.weightfile = weightfile
        if varlist is None:
            self.varlist = VAR_LIST_DEFAULT
        else:
            self.varlist = varlist
        self.filelist = self.get_clm_h0_filelist()
        self.filelist.sort()
        
    def get_clm_h0_filelist(self):
        print(f"{self.datapath}*.clm2.h0.*.nc")
        return glob.glob(f"{self.datapath}*.clm2.h0.*.nc")

    def make_annual_plots(self):
        pass
    
    def make_seasonal_plots(self, season):
        pass

    def make_all_plots_and_tables(self):
        pass

    def make_monthly_climatology_plots(self):
        pass

    def make_table_diagnostics(self):
        pass

    def find_case_year_range(self):
        year_start = int(self.filelist[0].split(".")[-2].split("-")[0])
        year_end = int(self.filelist[0].split(".")[-2].split("-")[0])
        files_missing = False
        if len(self.filelist) < (year_end - year_start)*12:
            files_missing = True
        return year_start, year_end, files_missing