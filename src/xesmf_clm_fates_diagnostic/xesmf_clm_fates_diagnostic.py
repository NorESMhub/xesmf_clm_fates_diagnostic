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

from .plotting_methods import make_se_regridder, regrid_se_data, make_bias_plot

# "FPSN", ,"BTRAN2" "SOILPSI","WA","QCHARGE","NEE","WT"
VAR_LIST_DEFAULT = ["TSA","RAIN","SNOW","TSOI","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","QINFL","QOVER","QRGWL","QDRAI","QINTR","QSOIL","QVEGT","SOILLIQ","SOILICE","SNOWLIQ","SNOWICE","ZWT","FCOV","PCO2", "landfrac","area","FSR","PBOT","SNOWDP","FSDS","FSA","FLDS","FIRE","FIRA","FSH","FCTR","FCEV","FGEV","FGR"]

VAR_LIST_CLIM = ["TSA","RAIN","SNOW","FSR","FSDS","FSA","FIRA","FCTR","FCEV","FGEV","QOVER","QDRAI","QRGWL","SNOWDP","FPSN","FSH","FSH_V","FSH_G","TV","TG","TSNOW","SABV","SABG","FIRE","FGR","FSM","TAUX","TAUY","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","BTRAN2","H2OSNO","H2OCAN","SNOWLIQ","SNOWICE","QINFL","QINTR","QDRIP","QSNOMELT","QSOIL","QVEGE","QVEGT","ERRSOI","ERRSEB","FSNO","ERRSOL","ERRH2O","TBOT","TLAKE","WIND","THBOT","QBOT","ZBOT","FLDS","FSDSNDLN","FSDSNI","FSDSVD","FSDSVDLN","FSDSVI","FSDSND","FSRND","FSRNDLN","FSRNI","FSRVD","FSRVDLN","FSRVI","Q2M","TREFMNAV","TREFMXAV","SOILLIQ","SOILICE","H2OSOI","TSOI","WA","WT","ZWT","QCHARGE","FCOV","PCO2","NEE","GPP","NPP","AR","HR","NEP","ER","SUPPLEMENT_TO_SMINN","SMINN_LEACHED","COL_FIRE_CLOSS","COL_FIRE_NLOSS","PFT_FIRE_CLOSS","PFT_FIRE_NLOSS","FIRESEASONL","FIRE_PROB","ANN_FAREA_BURNED","MEAN_FIRE_PROB","PBOT","SNOBCMCL","SNOBCMSL","SNODSTMCL","SNODSTMSL","SNOOCMCL","SNOOCMSL","BCDEP","DSTDEP","OCDEP"]

SEASONS = ["DJF", "MAM", "JJA", "SON"]

class XesmfCLMFatesDiagnostics:

    def __init__(self, datapath, weightfile, varlist=None, casename= None):
        self.datapath = datapath
        self.weightfile = weightfile
        # TODO: Cut varlist to variables actually in files
        if varlist is None:
            self.varlist = VAR_LIST_DEFAULT
        else:
            self.varlist = varlist
        self.filelist = self.get_clm_h0_filelist()
        self.filelist.sort()
        self.regridder = make_se_regridder(self.weightfile)
        if casename is None:
            self.casename = self.filelist[0].split("/")[-1].split(".")[0]
        
    def get_clm_h0_filelist(self):
        #print(f"{self.datapath}*.clm2.h0.*.nc")
        return glob.glob(f"{self.datapath}*.clm2.h0.*.nc")

    def make_annual_plots(self, year_range = None):

        outd = None
        for year in year_range:
            for month in range(12):
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine='netcdf4')[self.varlist]
                #print(outd_here)
                #sys.exit(4)
                if not outd:
                    outd = outd_here
                else: 
                    outd = xr.concat([outd, outd_here], dim= 'time')
        outd = outd.mean(dim="time")
        self.plot_all_the_variables_on_map(outd, year_range, plottype="annualmean")
  
    def plot_all_the_variables_on_map(self, outd, year_range, plottype):
      for var in self.varlist:
            if var in outd.keys():
                to_plot = regrid_se_data(self.regridder, outd[var])
                if 'levgrnd' in to_plot.dims or 'levsoi' in to_plot.dims:
                    to_plot = to_plot[0,:,:]
                make_bias_plot(to_plot, f"{self.casename}_{plottype}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}")


    def make_seasonal_plots(self, season, year_range):
        outd = None
        for year in year_range:
            for monthincr in range(3):

                month = monthincr + season*3
                if month == 0:
                    month = 12
                #print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine='netcdf4')[self.varlist]
                #print(outd_here)
                #sys.exit(4)
                if not outd:
                    outd = outd_here
                else: 
                    outd = xr.concat([outd, outd_here], dim= 'time')
        outd = outd.mean(dim="time")
        self.plot_all_the_variables_on_map(outd, year_range, plottype=SEASONS[season])

    def make_all_plots_and_tables(self, year_range = None):
        # TODO: Handle different year_range
        if year_range is None:
            year_range = self.get_year_range()
        self.make_annual_plots(year_range)
        for season in range(4):
            self.make_seasonal_plots(season, year_range)


    def get_year_range(self):
        year_start, year_end, files_missing = self.find_case_year_range()
        # TODO: Deal with missing files, also for different year_range
        if not files_missing:
            year_range = np.arange(max(year_start, year_end - 10), year_end +1)
        return year_range

    def make_monthly_climatology_plots(self):
        pass

    def make_table_diagnostics(self):
        pass

    def find_case_year_range(self):
        year_start = int(self.filelist[0].split(".")[-2].split("-")[0])
        year_end = int(self.filelist[-1].split(".")[-2].split("-")[0])
        files_missing = False
        if len(self.filelist) < (year_end - year_start)*12:
            files_missing = True
        return year_start, year_end, files_missing