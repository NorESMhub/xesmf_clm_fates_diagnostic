import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xarray as xr

import matplotlib.pyplot as plt

import os, sys, glob
#import netCDF4 as nc4
import shutil
import warnings
warnings.filterwarnings('ignore')

import xesmf as xe
import cartopy.crs as ccrs

from .plotting_methods import make_se_regridder, regrid_se_data, make_bias_plot
# "FPSN", ,"BTRAN2" "SOILPSI","WA","QCHARGE","NEE","WT"
VAR_LIST_DEFAULT = ["TSA","RAIN","SNOW","TSOI","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","QINFL","QOVER","QRGWL","QDRAI","QINTR","QSOIL","QVEGT","SOILLIQ","SOILICE","SNOWLIQ","SNOWICE","ZWT","FCOV","PCO2", "landfrac","area","FSR","PBOT","SNOWDP","FSDS","FSA","FLDS","FIRE","FIRA","FSH","FCTR","FCEV","FGEV","FGR"]

VAR_LIST_CLIM = ["TSA","RAIN","SNOW","FSR","FSDS","FSA","FIRA","FCTR","FCEV","FGEV","QOVER","QDRAI","QRGWL","SNOWDP","FPSN","FSH","FSH_V","FSH_G","TV","TG","TSNOW","SABV","SABG","FIRE","FGR","FSM","TAUX","TAUY","ELAI","ESAI","TLAI","TSAI","LAISUN","LAISHA","BTRAN2","H2OSNO","H2OCAN","SNOWLIQ","SNOWICE","QINFL","QINTR","QDRIP","QSNOMELT","QSOIL","QVEGE","QVEGT","ERRSOI","ERRSEB","FSNO","ERRSOL","ERRH2O","TBOT","TLAKE","WIND","THBOT","QBOT","ZBOT","FLDS","FSDSNDLN","FSDSNI","FSDSVD","FSDSVDLN","FSDSVI","FSDSND","FSRND","FSRNDLN","FSRNI","FSRVD","FSRVDLN","FSRVI","Q2M","TREFMNAV","TREFMXAV","SOILLIQ","SOILICE","H2OSOI","TSOI","WA","WT","ZWT","QCHARGE","FCOV","PCO2","NEE","GPP","NPP","AR","HR","NEP","ER","SUPPLEMENT_TO_SMINN","SMINN_LEACHED","COL_FIRE_CLOSS","COL_FIRE_NLOSS","PFT_FIRE_CLOSS","PFT_FIRE_NLOSS","FIRESEASONL","FIRE_PROB","ANN_FAREA_BURNED","MEAN_FIRE_PROB","PBOT","SNOBCMCL","SNOBCMSL","SNODSTMCL","SNODSTMSL","SNOOCMCL","SNOOCMSL","BCDEP","DSTDEP","OCDEP"]

MONTHS = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARSETS_TS = {"landf": ["TSA", "RAIN", "SNOW", "LAISUN"]}

class XesmfCLMFatesDiagnostics:

    def __init__(self, datapath, weightfile, varlist=None, casename= None, region_def = None):
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
            self.casename = ".".join(self.filelist[0].split("/")[-1].split(".")[:-4])
        else:
            self.casename = casename
        self.region_def = region_def
        
    def get_clm_h0_filelist(self):
        #print(f"{self.datapath}*.clm2.h0.*.nc")
        return glob.glob(f"{self.datapath}*.clm2.h0.*.nc")

    def get_annual_data(self, year_range = None, varlist= None):

        outd = None
        if varlist is None:
            varlist = self.varlist
        for year in year_range:
            for month in range(12):
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine='netcdf4')[varlist]
                #print(outd_here)
                #sys.exit(4)
                if not outd:
                    outd = outd_here
                else: 
                    outd = xr.concat([outd, outd_here], dim= 'time')
        outd = outd.mean(dim="time")
        return outd
  
    def plot_all_the_variables_on_map(self, outd, year_range, plottype):
      for var in self.varlist:
            if var in outd.keys():
                to_plot = regrid_se_data(self.regridder, outd[var])
                if 'levgrnd' in to_plot.dims or 'levsoi' in to_plot.dims:
                    to_plot = to_plot[0,:,:]
                make_bias_plot(to_plot, f"{self.casename}_{plottype}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}")


    def get_seasonal_data(self, season, year_range, varlist = None):
        outd = None
        if varlist is None:
            varlist = self.varlist
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
        return outd
    
    def get_monthly_climatology_data(self, year_range, varlist = None):
        outd_months = None
        if varlist is None:
            varlist = self.varlist
        for month in range(12):
            outd = None
            for year in year_range:
                #print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month+1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine='netcdf4')[self.varlist]
                #print(outd_here)
                #sys.exit(4)
                if not outd:
                    outd = outd_here
                else: 
                    outd = xr.concat([outd, outd_here], dim= 'time')
            outd = outd.mean(dim="time", keepdims=True)
            if outd_months is None:
                outd_months = outd
            else:
                outd_months = xr.concat([outd_months, outd], dim= 'time')
        return outd_months


    def make_all_plots_and_tables(self, year_range = None):
        # TODO: Handle different year_range
        if year_range is None:
            year_range = self.get_year_range()
        """
        outd = self.get_annual_data(year_range)

        self.plot_all_the_variables_on_map(outd, year_range, plottype="ANN")
        for season in range(4):
            outd = self.get_seasonal_data(season, year_range)
            self.plot_all_the_variables_on_map(outd, year_range, plottype=SEASONS[season])
        """
        for varsetname, varset in VARSETS_TS.items():
            self.make_all_regional_timeseries(year_range, varset, varsetname)

    def get_year_range(self):
        year_start, year_end, files_missing = self.find_case_year_range()
        # TODO: Deal with missing files, also for different year_range
        if not files_missing:
            year_range = np.arange(max(year_start, year_end - 10), year_end +1)
        return year_range
    
    def make_timeseries_plots_for_varlist(self, outd, varlist, regioninfo, figname):
        fig, axs = plt.subplots(ncols=2, nrows=int(np.ceil(len(varlist)/2)))
        for varnum, var in enumerate(varlist):
            outd_regr = regrid_se_data(self.regridder, outd[var])
            weights = np.cos(np.deg2rad(outd_regr.lat))
            weighted_data = outd_regr.weighted(weights)
            ts_data = weighted_data.mean(["lon", "lat"])
            axs[varnum//2, varnum%2].plot(range(12), ts_data)
            axs[varnum//2, varnum%2].set_xticks(ticks = range(12), labels = MONTHS)
            axs[varnum//2, varnum%2].set_title(var)
            # TODO set unit on y-axis
        fig.suptitle(f"{regioninfo[0]}, ({regioninfo[1]}) (yrs {regioninfo[2]})")
        plt.savefig(f"figs/{figname}.png")

    def make_all_regional_timeseries(self, year_range, varlist, varsetname):
        outd = self.get_monthly_climatology_data(year_range=year_range, varlist= varlist)
        self.make_timeseries_plots_for_varlist(
            outd, 
            varlist=varlist, 
            regioninfo=["Global", "global", f"{year_range[0]}-{year_range[1]}"], 
            figname=f"{self.casename}_{varsetname}_global"
            )
        # TODO: Fix this
        """
        if self.region_def:
            region_ds = xr.open_dataset(self.region_def)
            print(region_ds)
            for region in region_ds.region:
                cropped_ds = outd.sel(
                    lat=slice(region_ds.BOX_S[region],region_ds.BOX_N[region]), 
                    lon=slice(region_ds.BOX_W[region],region_ds.BOX_E[region])
                    )
                self.make_timeseries_plots_for_varlist(
                    cropped_ds,
                    varlist=varlist,
                    regioninfo= [region_ds.PTITSTR[region], region_ds.BOXSTR[region], f"{year_range[0]}-{year_range[1]}"],
                    figname=f"{self.casename}_{varsetname}_{region_ds.PTITSTR[region]}"
                )
        """
    def make_monthly_climatology_plots(self):
        pass

    def make_table_diagnostics(self):
        pass

    def make_combined_changeplots(self, other, variables, season="ANN", year_range= None):
        # TODO allow variable year_range
        if year_range is None:
            year_range = self.get_year_range()
            year_range_other = other.get_year_range()
            print(year_range_other)
            print(year_range)

            if year_range_other[0] < year_range[0]:
                year_range = year_range_other
            print(year_range)
            #sys.exit(4)     
        if season == "ANN":
            outd = self.get_annual_data(year_range, varlist = variables)
            outd_other = other.get_annual_data(year_range, varlist = variables)
            season_name = season
        else:
            outd = self.get_seasonal_data(season, year_range, varlist = None)
            outd_other = other.get_seasonal_data(season, year_range, varlist = None)
            season_name = SEASONS[season]        
        for var in variables:
            fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (10,15), subplot_kw={'projection': ccrs.PlateCarree()})
            to_plot = regrid_se_data(self.regridder, outd[var])
            to_plot_other = regrid_se_data(other.regridder, outd_other[var])
            make_bias_plot(to_plot,f"{self.casename} (yrs {year_range[0]}-{year_range[1]})",ax = axs[0])  
            make_bias_plot(to_plot_other,f"{other.casename} (yrs {year_range[0]}-{year_range[1]})",ax = axs[1])
            make_bias_plot(to_plot - to_plot_other,f"{self.casename} (yrs {year_range[0]}-{year_range[1]})",ax = axs[2])    
            # TODO: include units?
            fig.suptitle(f"{season_name} {var}")
            fig.savefig(f"figs/{self.casename}_compare_{other.casename}_{season_name}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}.png")

    def find_case_year_range(self):
        year_start = int(self.filelist[0].split(".")[-2].split("-")[0])
        year_end = int(self.filelist[-1].split(".")[-2].split("-")[0])
        files_missing = False
        if len(self.filelist) < (year_end - year_start)*12:
            files_missing = True
        return year_start, year_end, files_missing