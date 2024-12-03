import numpy as np

# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xarray as xr

import matplotlib.pyplot as plt

import os, sys, glob

# import netCDF4 as nc4
import shutil
import warnings

warnings.filterwarnings("ignore")

import xesmf as xe
import cartopy.crs as ccrs

from .plotting_methods import make_se_regridder, regrid_se_data, make_bias_plot

# "FPSN", ,"BTRAN2" "SOILPSI","WA","QCHARGE","NEE","WT"
VAR_LIST_DEFAULT = [
    "TSA",
    "RAIN",
    "SNOW",
    "TSOI",
    "ELAI",
    "ESAI",
    "TLAI",
    "TSAI",
    "LAISUN",
    "LAISHA",
    "QINFL",
    "QOVER",
    "QRGWL",
    "QDRAI",
    "QINTR",
    "QSOIL",
    "QVEGT",
    "SOILLIQ",
    "SOILICE",
    "SNOWLIQ",
    "SNOWICE",
    "ZWT",
    "FCOV",
    "PCO2",
    "landfrac",
    "area",
    "FSR",
    "PBOT",
    "SNOWDP",
    "FSDS",
    "FSA",
    "FLDS",
    "FIRE",
    "FIRA",
    "FSH",
    "FCTR",
    "FCEV",
    "FGEV",
    "FGR",
]

VAR_LIST_CLIM = [
    "TSA",
    "RAIN",
    "SNOW",
    "FSR",
    "FSDS",
    "FSA",
    "FIRA",
    "FCTR",
    "FCEV",
    "FGEV",
    "QOVER",
    "QDRAI",
    "QRGWL",
    "SNOWDP",
    "FPSN",
    "FSH",
    "FSH_V",
    "FSH_G",
    "TV",
    "TG",
    "TSNOW",
    "SABV",
    "SABG",
    "FIRE",
    "FGR",
    "FSM",
    "TAUX",
    "TAUY",
    "ELAI",
    "ESAI",
    "TLAI",
    "TSAI",
    "LAISUN",
    "LAISHA",
    "BTRAN2",
    "H2OSNO",
    "H2OCAN",
    "SNOWLIQ",
    "SNOWICE",
    "QINFL",
    "QINTR",
    "QDRIP",
    "QSNOMELT",
    "QSOIL",
    "QVEGE",
    "QVEGT",
    "ERRSOI",
    "ERRSEB",
    "FSNO",
    "ERRSOL",
    "ERRH2O",
    "TBOT",
    "TLAKE",
    "WIND",
    "THBOT",
    "QBOT",
    "ZBOT",
    "FLDS",
    "FSDSNDLN",
    "FSDSNI",
    "FSDSVD",
    "FSDSVDLN",
    "FSDSVI",
    "FSDSND",
    "FSRND",
    "FSRNDLN",
    "FSRNI",
    "FSRVD",
    "FSRVDLN",
    "FSRVI",
    "Q2M",
    "TREFMNAV",
    "TREFMXAV",
    "SOILLIQ",
    "SOILICE",
    "H2OSOI",
    "TSOI",
    "WA",
    "WT",
    "ZWT",
    "QCHARGE",
    "FCOV",
    "PCO2",
    "NEE",
    "GPP",
    "NPP",
    "AR",
    "HR",
    "NEP",
    "ER",
    "SUPPLEMENT_TO_SMINN",
    "SMINN_LEACHED",
    "COL_FIRE_CLOSS",
    "COL_FIRE_NLOSS",
    "PFT_FIRE_CLOSS",
    "PFT_FIRE_NLOSS",
    "FIRESEASONL",
    "FIRE_PROB",
    "ANN_FAREA_BURNED",
    "MEAN_FIRE_PROB",
    "PBOT",
    "SNOBCMCL",
    "SNOBCMSL",
    "SNODSTMCL",
    "SNODSTMSL",
    "SNOOCMCL",
    "SNOOCMSL",
    "BCDEP",
    "DSTDEP",
    "OCDEP",
]

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

SEASONS = ["DJF", "MAM", "JJA", "SON"]
VARSETS_TS = {"landf": ["TSA", "RAIN", "SNOW", "LAISUN"]}


class XesmfCLMFatesDiagnostics:
    """
    Class that holds and organises info on modelling outputs,
    regridding variables to plot up in diagnostics etc
    """
    def __init__(
        self, datapath, weightfile, varlist=None, casename=None, region_def=None, outdir = None
    ):
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
        if outdir is None:
            self.outdir = "figs/"
        else:
            self.outdir = outdir


    def get_clm_h0_filelist(self):
        """
        Get a list of all the files for the case

        Returns
        -------
        list
            with paths to all the clm2.h0 files
        """
        # print(f"{self.datapath}*.clm2.h0.*.nc")
        return glob.glob(f"{self.datapath}*.clm2.h0.*.nc")

    def get_annual_data(self, year_range, varlist=None):
        """
        Get annual mean data for variables in varlist

        Parameters
        ----------
        year_range : range
            Of years to include
        varlist : list
            List of variables to get data for, if not supplied, the objects
            varlist will be used. If the list includes variables not in the 
            outputfiles, they will be 
        """
        outd = None
        if varlist is None:
            varlist = self.varlist
        for year in year_range:
            for month in range(12):
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist]
                # print(outd_here)
                # sys.exit(4)
                if not outd:
                    outd = outd_here
                else:
                    outd = xr.concat([outd, outd_here], dim="time")
        outd = outd.mean(dim="time")
        return outd
    
    def get_annual_mean_ts(self, year_range, varlist=None):
        """
        Get annual mean data for variables in varlist

        Parameters
        ----------
        year_range : range
            Of years to include
        varlist : list
            List of variables to get data for, if not supplied, the objects
            varlist will be used. If the list includes variables not in the 
            outputfiles, they will be 
        """
        outd = None
        if varlist is None:
            varlist = self.varlist
        for year in year_range:
            outd_yr = None
            for month in range(12):                         
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist]
                # print(outd_here)
                # sys.exit(4)
                if not outd_yr:
                    outd_yr = outd_here
                else:
                    outd_yr = xr.concat([outd_yr, outd_here], dim="time")
            outd_yr = outd_yr.mean(dim="time")
            if not outd:
                outd = outd_yr
            else: 
                outd = xr.concat([outd, outd_yr], dim="time")
        return outd

    def plot_all_the_variables_on_map(self, outd, year_range, plottype):
        """
        Plot maps of all variables in varlist

        Parameters
        ----------
        outd : xr.Dataset
            Dataset on native grid, with single timestep
        year_range : np.ndarray
            Range of years this is the mean of, to annotate plot
        plottype : str
            Name of plottype to annotate the plot
        """
        for var in self.varlist:
            if var in outd.keys():
                to_plot = regrid_se_data(self.regridder, outd[var])
                if "levgrnd" in to_plot.dims or "levsoi" in to_plot.dims:
                    to_plot = to_plot[0, :, :]
                make_bias_plot(
                    to_plot,
                    f"{self.outdir}{self.casename}_{plottype}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}",
                )

    def get_seasonal_data(self, season, year_range, varlist=None):
        """
        Get climatological mean data for a season

        Parameters
        ----------
        season : int
            The season-number 0 is DJF, 1 is MAM, 2 is JJA
            and 3 is SON
        year_range : np.ndarray
            Range of years to include in climatological mean
        varlist : list
            List of variables to make plots of

        Returns
        -------
        xr.Dataset
            Of climatological seasonal means of the variables in varlist
            for the season requested
        """
        outd = None
        if varlist is None:
            varlist = self.varlist
        for year in year_range:
            for monthincr in range(3):

                month = monthincr + season * 3
                if month == 0:
                    month = 12
                # print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = (
                    f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month:02d}.nc"
                )
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[self.varlist]
                # print(outd_here)
                # sys.exit(4)
                if not outd:
                    outd = outd_here
                else:
                    outd = xr.concat([outd, outd_here], dim="time")
        outd = outd.mean(dim="time")
        return outd

    def get_monthly_climatology_data(self, year_range, varlist=None):
        outd_months = None
        if varlist is None:
            varlist = self.varlist
        for month in range(12):
            outd = None
            for year in year_range:
                # print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month+1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[self.varlist]
                # print(outd_here)
                # sys.exit(4)
                if not outd:
                    outd = outd_here
                else:
                    outd = xr.concat([outd, outd_here], dim="time")
            outd = outd.mean(dim="time", keepdims=True)
            if outd_months is None:
                outd_months = outd
            else:
                outd_months = xr.concat([outd_months, outd], dim="time")
        return outd_months

    def make_all_plots_and_tables(self, year_range=None):
        # TODO: Handle different year_range
        self.make_global_yearly_trends(year_range)

        if year_range is None:
            year_range = self.get_year_range()
        outd = self.get_annual_data(year_range)

        self.plot_all_the_variables_on_map(outd, year_range, plottype="ANN")
        for season in range(4):
            outd = self.get_seasonal_data(season, year_range)
            self.plot_all_the_variables_on_map(
                outd, year_range, plottype=SEASONS[season]
            )
        for varsetname, varset in VARSETS_TS.items():
            self.make_all_regional_timeseries(year_range, varset, varsetname)

    def get_year_range(self):
        year_start, year_end, files_missing = self.find_case_year_range()
        # TODO: Deal with missing files, also for different year_range
        if not files_missing:
            year_range = np.arange(max(year_start, year_end - 10), year_end + 1)
        return year_range

    def make_timeseries_plots_for_varlist(
        self, outd, varlist, region_df, year_range_string, varsetname
    ):
        figs = []
        for i in range(region_df.shape[0]):
            fig, axs = plt.subplots(ncols=2, nrows=int(np.ceil(len(varlist) / 2)))
            figs.append([fig, axs])
        for varnum, var in enumerate(varlist):
            outd_regr = regrid_se_data(self.regridder, outd[var])
            for region, region_info in region_df.iterrows():
                crop = outd_regr.sel(
                    lat=slice(region_info["BOX_S"], region_info["BOX_N"]),
                    lon=slice(region_info["BOX_W"], region_info["BOX_E"]),
                )
                weights = np.cos(np.deg2rad(crop.lat))
                weighted_data = crop.weighted(weights)
                ts_data = weighted_data.mean(["lon", "lat"])
                figs[region][1][varnum // 2, varnum % 2].plot(range(12), ts_data)
                figs[region][1][varnum // 2, varnum % 2].set_xticks(
                    ticks=range(12), labels=MONTHS
                )
                figs[region][1][varnum // 2, varnum % 2].set_title(var)
                # print(ts_data)
                # figs[region][1][varnum//2, varnum%2].set_ylabel(ts_data.unit)
            # TODO set unit on y-axis
        for region, region_info in region_df.iterrows():
            figs[region][0].suptitle(
                f"{region_info['PTITSTR']}, ({region_info['BOXSTR']}) (yrs {year_range_string})"
            )
            figs[region][0].tight_layout()
            figs[region][0].savefig(
                f"{self.outdir}{self.casename}_{varsetname}_{region_info['PTITSTR'].replace(' ', '')}.png"
            )

    def make_all_regional_timeseries(self, year_range, varlist, varsetname):
        if self.region_def:
            region_ds = xr.open_dataset(self.region_def)
            region_df = region_ds.to_dataframe()
            region_df = region_df.reindex(
                index=region_df.index[::-1]
            )  # , inplace=True)
            region_df = region_df.astype({"PTITSTR": str, "BOXSTR": str})
            print(region_df)
            print(region_df.shape)
        else:
            region_df = pd.Dataset(
                data={
                    "BOX_S": -100.0,
                    "BOX_N": 100.0,
                    "BOX_W": -200.0,
                    "BOX_E": 200.0,
                    "PS_ID": "Global",
                    "PTITSTR": "Global",
                    "BOXSTR": "(90S-90N,180W-180E)",
                }
            )
        outd = self.get_monthly_climatology_data(year_range=year_range, varlist=varlist)
        self.make_timeseries_plots_for_varlist(
            outd,
            varlist=varlist,
            region_df=region_df,
            year_range_string=f"{year_range[0]}-{year_range[-1]}",
            varsetname=varsetname,
        )

    def make_global_yearly_trends(self, varlist = None, year_range = None):
        if varlist is None:
            varlist = self.varlist
        if year_range is not None:
            yr_start = year_range[0]
            yr_end = year_range[1]
        else:
            yr_start, yr_end, missing = self.find_case_year_range()
            year_range =np.arange(yr_start, yr_end + 1)
        if yr_end == yr_start:
            print("Can not make global annual trend plots from just one year of data")
            return
   
        ts_data = np.zeros((len(varlist), len(year_range)))
        weights = None
        
        if not missing:
            outd = self.get_annual_mean_ts(year_range, varlist=varlist)
            for varnum, var in enumerate(varlist):
                outd_regr = regrid_se_data(self.regridder, outd[var])
                if weights is None:
                    weights = np.cos(np.deg2rad(outd_regr.lat))
                weighted = outd_regr.weighted(weights)
                if len(outd_regr.values.shape) > 3:
                    ts_data[varnum, :] = weighted.mean(["lon", "lat"]).values[:,0]
                else: 
                    ts_data[varnum, :] = weighted.mean(["lon", "lat"]).values
        fig_count = 0
        fig, axs = plt.subplots(ncols = 5, nrows= 5, figsize=(30,30))
        #fig.suptitle("Global annual trends")
        for varnum, var in enumerate(varlist):
            if varnum%25 == 0 and varnum > 0:
                fig.tight_layout()
                fig.savefig(f"{self.outdir}{self.casename}_glob_ann_trendplot_num{fig_count}_{yr_start}-{yr_end}.png")
                plt.clf()
                fig, axs = plt.subplots(ncols = 5, nrows= 5, figsize=(30,30))
                fig_count = fig_count + 1
            axs[(varnum%25)//5, (varnum%25)%5].plot(year_range, ts_data[varnum, :])
            axs[(varnum%25)//5, (varnum%25)%5].set_title(var, size=30)
            axs[(varnum%25)//5, (varnum%25)%5].set_xlabel("Year", size=25)
        fig.tight_layout()
        fig.savefig(f"{self.outdir}{self.casename}_glob_ann_trendplot_num{fig_count}_{yr_start}-{yr_end}.png")
        plt.clf()
        return        
        #for year in find_case_year_range(self):


    def make_table_diagnostics(self):
        pass

    def make_combined_changeplots(
        self, other, variables, season="ANN", year_range=None
    ):
        # TODO allow variable year_range
        if year_range is None:
            year_range = self.get_year_range()
            year_range_other = other.get_year_range()
            print(year_range_other)
            print(year_range)

            if year_range_other[0] < year_range[0]:
                year_range = year_range_other
            print(year_range)
            # sys.exit(4)
        if season == "ANN":
            outd = self.get_annual_data(year_range, varlist=variables)
            outd_other = other.get_annual_data(year_range, varlist=variables)
            season_name = season
        else:
            outd = self.get_seasonal_data(season, year_range, varlist=None)
            outd_other = other.get_seasonal_data(season, year_range, varlist=None)
            season_name = SEASONS[season]
        for var in variables:
            fig, axs = plt.subplots(
                nrows=3,
                ncols=1,
                figsize=(10, 15),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            to_plot = regrid_se_data(self.regridder, outd[var])
            to_plot_other = regrid_se_data(other.regridder, outd_other[var])
            make_bias_plot(
                to_plot,
                f"{self.outdir}{self.casename} (yrs {year_range[0]}-{year_range[1]})",
                ax=axs[0],
            )
            make_bias_plot(
                to_plot_other,
                f"{self.outdir}{other.casename} (yrs {year_range[0]}-{year_range[1]})",
                ax=axs[1],
            )
            make_bias_plot(
                to_plot - to_plot_other,
                f"{self.outdir}{self.casename} (yrs {year_range[0]}-{year_range[1]})",
                ax=axs[2],
            )
            # TODO: include units?
            fig.suptitle(f"{season_name} {var}")
            fig.savefig(
                f"{self.outdir}{self.casename}_compare_{other.casename}_{season_name}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}.png"
            )

    def find_case_year_range(self):
        year_start = int(self.filelist[0].split(".")[-2].split("-")[0])
        year_end = int(self.filelist[-1].split(".")[-2].split("-")[0])
        files_missing = False
        if len(self.filelist) < (year_end - year_start) * 12:
            files_missing = True
        return year_start, year_end, files_missing
