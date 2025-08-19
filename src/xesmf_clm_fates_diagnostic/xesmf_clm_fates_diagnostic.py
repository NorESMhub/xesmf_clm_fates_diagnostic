import numpy as np

import xarray as xr

import matplotlib.pyplot as plt

import os, sys, glob

# import netCDF4 as nc4
import warnings

warnings.filterwarnings("ignore")

import cartopy.crs as ccrs

from .plotting_methods import make_generic_regridder, regrid_se_data, make_bias_plot, make_regridder_regular_to_coarsest_resolution
from .infrastructure_help_functions import setup_nested_folder_structure_from_dict, read_pam_file#, clean_empty_folders_in_tree
from  .misc_help_functions import get_unit_conversion_and_new_label, make_regridding_target_from_weightfile, get_unit_conversion_from_string, do_light_unit_string_conversion, SEASONS, calculate_rmse_from_bias

MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


#def get_minimal_intersecting_year_range(year_range, year_range_other):



class XesmfCLMFatesDiagnostics:
    """
    Class that holds and organises info on modelling outputs,
    regridding variables to plot up in diagnostics etc
    """
    def __init__(
        self, datapath, weightfile, pamfile, casename=None, region_def=None, outdir = None,
    ):
        self.datapath = datapath
        self.weightfile = weightfile
        self.var_pams = read_pam_file(pamfile)
        print(self.var_pams)
        #sys.exit(4)
        self.filelist = self.get_clm_h0_filelist()
        self.filelist.sort()
        self.regridder = make_generic_regridder(self.weightfile, self.filelist[0])
        self.regrid_target = make_regridding_target_from_weightfile(self.weightfile, self.filelist[0])
        if casename is None:
            self.casename = ".".join(self.filelist[0].split("/")[-1].split(".")[:-4])
        else:
            self.casename = casename
        self.region_def = region_def
        if outdir is None:
            outdir = "figs/"
        self.setup_folder_structure(outdir)
        self.unit_dict = {}
        self.set_composite_variable_dict()
        vars_missing = self.wet_and_cut_varlists()
        if len(vars_missing):
            print("Not all requested variables are available in output, ignoring these:")
            print(vars_missing)
        print(self.var_pams)
        print(self.unit_dict)
        #sys.exit(4)
        #sys.exit(4)

    # TODO: Make this not hard-coded
    def set_composite_variable_dict(self):
        self.composite_variable_dict = {
            "albedo": [["FSR", "FSDS"], "divide"],
            "pr": [["RAIN", "SNOW"], "add"],
            "npp_gpp_fraction":[["FATES_NPP", "FATES_GPP"], "divide"]
            }

    def wet_and_cut_varlists(self):
        # TODO: Make sure all items in SEASONAL is also in var_list_main 
        read = xr.open_dataset(self.filelist[0]).keys()
        lists_check =["VAR_LIST_MAIN", "COMPARE_VARIABLES"]

        vars_missing = []
        vars_needed_for_composites = []
        for list_n in lists_check:
            for item in self.var_pams[list_n]:
                if item not in read:
                    if item in self.composite_variable_dict:
                        composit_works = True
                        for comp_item in self.composite_variable_dict[item][0]:
                            print(f"Doing composite item {item}")
                            if comp_item not in read:
                                composit_works = False
                                break
                        if composit_works:
                            vars_needed_for_composites.extend(self.composite_variable_dict[item][0])
                        else:
                          vars_missing.append(item)  
                    else:
                        vars_missing.append(item)
            self.var_pams[list_n] = list(set(self.var_pams[list_n]) - set(vars_missing))
            self.var_pams[list_n] = list(set(self.var_pams[list_n]).union(set(vars_needed_for_composites)))

        for varsetname, varset in self.var_pams["SEASONAL_VARSETS"].items():
            for vari in varset:
                if vari not in read and vari not in vars_missing:
                    vars_missing.append(vari)
        for varset in self.var_pams["SEASONAL_VARSETS"]:
            self.var_pams["SEASONAL_VARSETS"][varset] = list(set(self.var_pams["SEASONAL_VARSETS"][varset]) - set(vars_missing))
        return vars_missing

    def setup_folder_structure(self, outdir):
        if not os.path.exists(outdir):
            raise ValueError(f"{outdir} must be an existing directory")
        
        subfolder_structure = {
            f"{self.casename}": {
                "trends": None, 
                "clim_maps": ["ANN", "DJF", "MAM", "JJA", "SON"], 
                "seasonal_cycle": None, 
            }
        }
        
        setup_nested_folder_structure_from_dict(outdir, subfolder_structure)
        self.outdir = f"{outdir}/{self.casename}"

    
    def setup_folders_for_comparison_plots(self, other, season):
        subfolder_structure = {
            "compare": {other.casename: season}
        }
        setup_nested_folder_structure_from_dict(self.outdir, subfolder_structure)
        return f"{self.outdir}/compare/{other.casename}/{season}"
    
    def setup_folders_for_seasonal_cycle_plots(self, varset):
        subfolder_structure = {
            "seasonal_cycle": varset
        }
        setup_nested_folder_structure_from_dict(self.outdir, subfolder_structure)
        return
    
    def setup_folders_for_observation_plots(self, season):
        subfolder_structure = {
            "OBS_comparison": season
        }
        setup_nested_folder_structure_from_dict(self.outdir, subfolder_structure)
        return f"{self.outdir}/OBS_comparison/{season}"       


    #def clean_out_empty_folders(self):
    #    clean_empty_folders_in_tree(self.outdir)

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
    
    def add_composite_variables(self, outd, varlist_composite):
        """
        Add composite variables to outd using existing variables

        Parameters
        ----------
        outd : xr.Dataset
            Dataset of non-composite variables
        varlist_composite : list
            List of composite variables
        """
        for composite in varlist_composite:
            if self.composite_variable_dict[composite][1] == "add":
                outd[composite] = outd[self.composite_variable_dict[composite][0][0]]
                for i in range(len(self.composite_variable_dict[composite][0]) - 1):
                    outd[composite] = outd[composite] + outd[self.composite_variable_dict[composite][0][i+1]]
            if self.composite_variable_dict[composite][1] == "divide":
                outd[composite] = outd[self.composite_variable_dict[composite][0][0]] / outd[self.composite_variable_dict[composite][0][1]]
        return outd
    

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
            varlist = self.var_pams["VAR_LIST_MAIN"]
        varlist_direct = list(set(varlist) - set(self.composite_variable_dict.keys()))
        varlist_composite = set(varlist).intersection(self.composite_variable_dict.keys())
        
        for year in year_range:
            for month in range(12):
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist_direct]
                outd_here = self.add_composite_variables(outd_here, varlist_composite)
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
            varlist = self.var_pams["VAR_LIST_MAIN"]
        varlist_direct = list(set(varlist) - set(self.composite_variable_dict.keys()))
        varlist_composite = set(varlist).intersection(self.composite_variable_dict.keys())

        for year in year_range:
            outd_yr = None
            for month in range(12):                         
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist_direct]
                outd_here = self.add_composite_variables(outd_here, varlist_composite)
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

        self.add_to_unit_dict(self.var_pams["VAR_LIST_MAIN"])
        for var in self.var_pams["VAR_LIST_MAIN"]:
            if var in outd.keys():
                logscale = False
                if "LOG_PLOT" in self.var_pams and var in self.var_pams["LOG_PLOT"]:
                    logscale = True
                to_plot = regrid_se_data(self.regridder, outd[var])
                if "levgrnd" in to_plot.dims or "levsoi" in to_plot.dims:
                    to_plot = to_plot[0, :, :]
                make_bias_plot(
                    to_plot,
                    f"{self.outdir}/clim_maps/{plottype}/{self.casename}_{plottype}_{var}_{year_range[0]:04d}-{year_range[-1]:04d}",
                    xlabel = f"{plottype} {var} [{self.unit_dict[var]}]",
                    logscale=logscale
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
            varlist = self.var_pams["VAR_LIST_MAIN"]
        varlist_direct = list(set(varlist) - set(self.composite_variable_dict.keys()))
        varlist_composite = set(varlist).intersection(self.composite_variable_dict.keys())
        
        for year in year_range:
            for monthincr in range(3):

                month = monthincr + season * 3
                if month == 0:
                    month = 12
                # print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = (
                    f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month:02d}.nc"
                )
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist_direct]
                outd_here = self.add_composite_variables(outd_here, varlist_composite)
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
            varlist = self.var_pams["VAR_LIST_MAIN"]
        varlist_direct = list(set(varlist) - set(self.composite_variable_dict.keys()))
        varlist_composite = set(varlist).intersection(self.composite_variable_dict.keys())
        for month in range(12):
            outd = None
            for year in year_range:
                # print(f"Season: {season}, monthincr: {monthincr}, month: {monthincr}")
                mfile = f"{self.datapath}/{self.casename}.clm2.h0.{year:04d}-{month+1:02d}.nc"
                outd_here = xr.open_dataset(mfile, engine="netcdf4")[varlist_direct]
                outd_here = self.add_composite_variables(outd_here, varlist_composite)
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

    def make_all_plots_and_tables(self, year_range=None, ilamb_cfgs = None):
        # TODO: Handle different year_range
        if year_range is None:
            year_range = self.get_year_range()
            year_range_full  = self.find_case_year_range()
            year_range_full = np.arange(year_range_full[0], year_range_full[1])
        
        self.make_global_yearly_trends(year_range=year_range_full)
        outd = self.get_annual_data(year_range)

        self.plot_all_the_variables_on_map(outd, year_range, plottype="ANN")
        for season in range(4):
            outd = self.get_seasonal_data(season, year_range)
            self.plot_all_the_variables_on_map(
                outd, year_range, plottype=SEASONS[season]
            )
        for varsetname, varset in self.var_pams["SEASONAL_VARSETS"].items():

            self.make_all_regional_timeseries(year_range, varset, varsetname, ilamb_cfgs=ilamb_cfgs)

    def get_year_range(self):
        year_start, year_end, files_missing = self.find_case_year_range()
        # TODO: Deal with missing files, also for different year_range
        if not files_missing:
            year_range = np.arange(max(year_start, year_end - 20), year_end + 1)
        else:
            raise ValueError(f"Files are missing in the year range from {year_start}, {year_end}") 
        return year_range
    
    def setup_seasonal_cycle_plots_and_add_data_for_varlist(
            self, varlist, region_df, year_range_string, varsetname, ilamb_cfgs
    ):
        figs = []
        rownum = int(np.ceil(len(varlist) / 2))        
        for i in range(region_df.shape[0]):
            fig, axs = plt.subplots(ncols=2, nrows=rownum)
            figs.append([fig, axs])
        print(varlist)
        for varnum, var in enumerate(varlist):
            for region, region_info in region_df.iterrows():
                if rownum < 2:
                    axnow = figs[region][1][varnum % 2]
                else: 
                    axnow = figs[region][1][varnum // 2, varnum % 2]
                axnow.set_xticks(ticks=range(12), labels=MONTHS)
                axnow.set_title(var)
                axnow.set_ylabel(ylabel)

        # Add data to plots
        for region, region_info in region_df.iterrows():
            figs[region][0].suptitle(
                f"{region_info['PTITSTR']}, ({region_info['BOXSTR']}) (yrs {year_range_string})",
                size = "x-large"
            )
            figs[region][0].tight_layout()
            figs[region][0].savefig(
                f"{self.outdir}/seasonal_cycle/{varsetname}/{self.casename}_{varsetname}_{region_info['PTITSTR'].replace(' ', '')}.png"
            )

    def make_timeseries_plots_for_varlist(
        self, outd, varlist, region_df, year_range_string, varsetname, ilamb_cfgs = None
    ):
        print("making regional seasonal cycle plots")
        print(ilamb_cfgs)
        self.add_to_unit_dict(varlist)
        figs = []
        rownum = int(np.ceil(len(varlist) / 2))
        print(varlist)
        for i in range(region_df.shape[0]):
            fig, axs = plt.subplots(ncols=2, nrows=rownum)
            figs.append([fig, axs])
        print(varlist)
        for varnum, var in enumerate(varlist):
            if ilamb_cfgs is None:
                unit_conversion_factor = 1
                unit_to_print = self.unit_dict[var]
            else: 
                unit_conversion_factor, unit_to_print = get_unit_conversion_from_string(ilamb_cfgs.get_variable_plot_unit(var), self.unit_dict[var])
            shift, ylabel = get_unit_conversion_and_new_label(unit_to_print)
            outd_regr = unit_conversion_factor * regrid_se_data(self.regridder, outd[var])
            for region, region_info in region_df.iterrows():
                if rownum < 2:
                    axnow = figs[region][1][varnum % 2]
                else: 
                    axnow = figs[region][1][varnum // 2, varnum % 2]

                crop = outd_regr.sel(
                    lat=(outd_regr.lat >= region_info["BOX_S"]) & (outd_regr.lat <= region_info["BOX_N"])
                    #lon=slice(region_info["BOX_W"], region_info["BOX_E"]),
                )
                if region_info["BOX_W"]%360 > region_info["BOX_E"]%360:
                    crop = crop.sel(lon=(crop.lon%360 >= region_info["BOX_W"]%360) | (crop.lon%360 <=region_info["BOX_E"]%360))
                elif region_info["BOX_W"]%360 == region_info["BOX_E"]%360:
                        crop = crop
                else:
                    crop = crop.sel(lon=(crop.lon%360 >= region_info["BOX_W"]%360) & (crop.lon%360 <=region_info["BOX_E"]%360)) 

                weights = np.cos(np.deg2rad(crop.lat))
                weighted_data = crop.weighted(weights)
                ts_data = weighted_data.mean(["lon", "lat"])

                
                axnow.plot(range(12), ts_data + shift, label = "mod")
                axnow.set_xticks(ticks=range(12), labels=MONTHS)
                axnow.set_title(var)
                axnow.set_ylabel(ylabel)
        if (ilamb_cfgs is not None) and (self.var_pams["OBSERVATION_COMPARISON"] is not None):
            print("Adding ilamb observations")
            ilamb_cfgs.add_seasonal_obsdata_to_axis(figs, varlist, region_df, self.var_pams["OBSERVATION_COMPARISON"])
        for region, region_info in region_df.iterrows():
            figs[region][0].suptitle(
                f"{region_info['PTITSTR']}, ({region_info['BOXSTR']}) (yrs {year_range_string})",
                size = "xx-large"
            )
            figs[region][0].tight_layout()
            axnow.legend()
            figs[region][0].savefig(
                f"{self.outdir}/seasonal_cycle/{varsetname}/{self.casename}_{varsetname}_{region_info['PTITSTR'].replace(' ', '')}.png"
            )

    def make_all_regional_timeseries(self, year_range, varlist, varsetname, ilamb_cfgs= None):
        self.setup_folders_for_seasonal_cycle_plots(varsetname)
        if self.region_def:
            region_ds = xr.open_dataset(self.region_def)
            region_df = region_ds.to_dataframe()
            region_df = region_df.reindex(
                index=region_df.index[::-1]
            )  # , inplace=True)
            region_df = region_df.astype({"PTITSTR": str, "BOXSTR": str})
        else:
            region_df = pd.Dataset(
                data={
                    "BOX_S": -100.0,
                    "BOX_N": 100.0,
                    "BOX_W": -180.0,
                    "BOX_E": 180.0,
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
            ilamb_cfgs = ilamb_cfgs
        )
        # add other models?

        # Add observations

    def make_global_yearly_trends(self, varlist = None, year_range = None):
        if varlist is None:
            varlist = self.var_pams["VAR_LIST_MAIN"]
        if year_range is not None:
            yr_start = year_range[0]
            yr_end = year_range[1]
            missing = False
        else:
            yr_start, yr_end, missing = self.find_case_year_range()
            year_range =np.arange(yr_start, yr_end + 1)
        if yr_end == yr_start:
            print("Can not make global annual trend plots from just one year of data")
            return
        self.add_to_unit_dict(varlist)
        ts_data = np.zeros((len(varlist), len(year_range)))
        weights = None
        
        if not missing:
            print(varlist)
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
                fig.savefig(f"{self.outdir}/trends/{self.casename}_glob_ann_trendplot_num{fig_count}_{yr_start}-{yr_end}.png")
                plt.clf()
                fig, axs = plt.subplots(ncols = 5, nrows= 5, figsize=(30,30))
                fig_count = fig_count + 1
            # TODO: Make this more general to handle other unit changes
            shift, ylabel = get_unit_conversion_and_new_label(self.unit_dict[var])
            axs[(varnum%25)//5, (varnum%25)%5].plot(year_range, ts_data[varnum, :]+ shift)
            axs[(varnum%25)//5, (varnum%25)%5].set_title(var, size=30)
            axs[(varnum%25)//5, (varnum%25)%5].set_ylabel(ylabel, size=25)
            axs[(varnum%25)//5, (varnum%25)%5].set_xlabel("Year", size=25)
        fig.tight_layout()
        fig.savefig(f"{self.outdir}/trends/{self.casename}_glob_ann_trendplot_num{fig_count}_{yr_start}-{yr_end}.png")
        plt.clf()
        return        
        #for year in find_case_year_range(self):


    def make_table_diagnostics(self):
        pass
    
    def get_year_ranges_for_comparison(self, other, year_range_in=None):
        year_range_avail = self.get_year_range()
        year_range_other_avail = other.get_year_range()
        if year_range_in is None:
            year_range = year_range_avail
            year_range_other = year_range_other_avail
            if year_range_other[0] < year_range[0]:
                year_range = year_range_other
            else:
                year_range_other = year_range
            return year_range, year_range_other
        
        if "year_range" in year_range_in:
            year_range = year_range_in["year_range"]
            if "year_range_other" in year_range_in:
                year_range_other = year_range_in["year_range_other"]
            else:
                year_range_other = year_range
            if year_range_other[0] < year_range_other_avail[0] or year_range_other[-1] > year_range_other_avail[-1]:
                year_range_other = np.arange(np.max(year_range_other[0], year_range_other_avail[0]))
        elif "compare_from_start" in year_range_in:
            year_range = np.arange(year_range_avail[0], year_range_avail[0] + year_range_in["compare_from_start"])
            year_range_other = np.arange(year_range_other_avail[0], year_range_other_avail[0] + year_range_in["compare_from_start"])
        elif "compare_from_end" in year_range_in:
            year_range = np.arange(year_range_avail[-1]-year_range_in["compare_from_end"], year_range_avail[-1]+1)
            year_range_other = np.arange(year_range_other_avail[-1]-year_range_in["compare_from_end"], year_range_other_avail[-1]+1)
              
        try:
            year_range = np.arange(np.max((year_range[0], year_range_avail[0])), np.min((year_range[-1], year_range_avail[-1])) + 1)
            year_range_other = np.arange(np.max((year_range_other[0], year_range_other_avail[0])), np.min((year_range_other[-1], year_range_other_avail[-1])) + 1)
        except:
            raise ValueError(f"The requested year ranges {year_range} and {year_range_other} have no overlap with available data ranges {year_range_avail} and {year_range_other_avail}")
        return year_range, year_range_other


    def make_combined_changeplots(
        self, other, variables=None, season="ANN", year_range_in=None, ilamb_cfgs = None
    ):
        fig_dir = self.setup_folders_for_comparison_plots(other, season)
        if variables is None:
            variables = list(set(self.var_pams["COMPARE_VARIABLES"]).intersection(set(other.var_pams["COMPARE_VARIABLES"])))
        self.add_to_unit_dict(variables)
        year_range, year_range_other = self.get_year_ranges_for_comparison(other, year_range_in)
        regridder_between, regrid_self_to_other = make_regridder_regular_to_coarsest_resolution(self.regrid_target, other.regrid_target)

        if season == "ANN":
            outd = self.get_annual_data(year_range, varlist=variables)
            outd_other = other.get_annual_data(year_range_other, varlist=variables)
            season_name = season
            fig_dir = self.setup_folders_for_comparison_plots(other, season)
        else:
            outd = self.get_seasonal_data(season, year_range, varlist=None)
            outd_other = other.get_seasonal_data(season, year_range_other, varlist=None)
            season_name = SEASONS[season]
            fig_dir = self.setup_folders_for_comparison_plots(other, SEASONS[season])
        for var in variables:
            if ilamb_cfgs is None:
                 unit_conversion_factor = 1
                 unit_to_print = self.unit_dict[var]
            else:
                unit_conversion_factor, unit_to_print = get_unit_conversion_from_string(ilamb_cfgs.get_variable_plot_unit(var), self.unit_dict[var])
            #print(f"{var} has {unit_conversion_factor} and to get {unit_to_print}")
            logscale = False
            if "LOG_PLOT" in self.var_pams and var in self.var_pams["LOG_PLOT"]:
                logscale = True
            fig, axs = plt.subplots(
                nrows=1,
                ncols=3,
                figsize=(30, 10),
                subplot_kw={"projection": ccrs.Robinson()},
                layout = 'constrained'
            )
            # Regridding block
            to_plot = regrid_se_data(self.regridder, outd[var]) * unit_conversion_factor
            to_plot_other = regrid_se_data(other.regridder, outd_other[var]) * unit_conversion_factor
            if regridder_between is not None:
                if regrid_self_to_other:
                    to_plot = regridder_between(to_plot)
                else:
                    to_plot_other = regridder_between(to_plot_other)
            if ilamb_cfgs is None:
                ymaxv = np.max((to_plot.max(), to_plot_other.max()))
                yminv = np.max((to_plot.min(), to_plot_other.min()))
                diffrange = None
                negdiffrange = None
            elif var in ilamb_cfgs.configurations:
                 yminv, ymaxv, diffrange, negdiffrange = ilamb_cfgs.configurations[var].obs_limits
            else:
                ymaxv = np.max((to_plot.max(), to_plot_other.max()))
                yminv = np.max((to_plot.min(), to_plot_other.min()))
                diffrange = None
                negdiffrange = None                
            if year_range[0] == year_range_other[0] and year_range[-1] == year_range_other[-1]:
                year_range_str = f"{year_range[0]:04d}-{year_range[-1]:04d}"
            else:
                year_range_str = f"{year_range[0]:04d}-{year_range[-1]:04d}_vs_{year_range_other[0]:04d}-{year_range_other[-1]:04d}"
            make_bias_plot(
                to_plot,
                f"{self.casename}",
                ax=axs[0],
                yminv = yminv,
                ymaxv = ymaxv,
                xlabel = f"{season} {var} [{unit_to_print}]",
                logscale=logscale
            )
            make_bias_plot(
                to_plot_other,
                f"{other.casename}",
                ax=axs[1],
                yminv = yminv,
                ymaxv = ymaxv,
                xlabel = f"{season} {var} [{unit_to_print}]",
                logscale=logscale
            )
            make_bias_plot(
                to_plot - to_plot_other,
                f"{self.casename} - {other.casename}",
                ax=axs[2], 
                cmap = "RdYlBu_r",
                yminv = negdiffrange,
                ymaxv = diffrange,
            )
            rmse = calculate_rmse_from_bias(to_plot - to_plot_other)
            fig.suptitle(f"{season_name} {var} ({self.unit_dict[var]}) (years {year_range_str}), RMSE = {rmse}", size = "xx-large", y=0.8)
            fig.savefig(
                f"{fig_dir}/{self.casename}_compare_{other.casename}_{season_name}_{var}_{year_range_str}.png"
            )
        if regridder_between is None:
            return
        regridder_between.grid_in.destroy()
        regridder_between.grid_out.destroy()
        del regridder_between


    def find_case_year_range(self):
        year_start = int(self.filelist[0].split(".")[-2].split("-")[0])
        year_end = int(self.filelist[-1].split(".")[-2].split("-")[0])
        files_missing = False
        if len(self.filelist) < (year_end - year_start) * 12:
            files_missing = True
            print(len(self.filelist))
            print((year_end - year_start) * 12)
            print(self.filelist)
        return year_start, year_end, files_missing

    def add_to_unit_dict(self, varlist):
        print(varlist)
        print(self.unit_dict)
        missing = list(set(varlist) - set(self.unit_dict.keys()))
        if len(missing) < 1:
            return
        read = xr.open_dataset(self.filelist[0])
        for vrm in missing:
            if vrm in read.keys():
                if "units" in read[vrm].attrs.keys():
                    self.unit_dict[vrm] = do_light_unit_string_conversion(read[vrm].attrs["units"])
                else:
                    self.unit_dict[vrm] = "No unit"
            elif vrm in self.composite_variable_dict.keys():
                factor1 = self.composite_variable_dict[vrm][0][0]
                if self.composite_variable_dict[vrm][1] == "add":
                    if factor1 in read.keys():
                        if "units" in read[factor1].attrs.keys():
                            self.unit_dict[vrm] = do_light_unit_string_conversion(read[factor1].attrs["units"])
                        else:
                            self.unit_dict[vrm] = "No unit"
                if self.composite_variable_dict[vrm][1] == "divide":
                    factor2 = self.composite_variable_dict[vrm][0][1]
                    if factor1 in read.keys() and factor2 in read.keys():
                        if "units" in read[factor1].attrs.keys() and "units" in read[factor2].attrs.keys():
                            unit1 = read[factor1].attrs["units"]
                            unit2 = read[factor2].attrs["units"]
                            if unit1 == unit2:
                                self.unit_dict[vrm] = "No unit"
                            else:
                                self.unit_dict[vrm] = do_light_unit_string_conversion(f"{unit1}/{unit2}")
                        else:
                            self.unit_dict[vrm] = "No unit"                         



    def make_alternate_varlist(self, ilamb_cfgs, variables):
        varlist = []
        to_remove = []
        for var in variables:
            print(var)
            varname_mod = ilamb_cfgs.get_varname_in_file(var, self.var_pams["VAR_LIST_MAIN"])
            if varname_mod is not None:
                varlist.append(varname_mod)
            else:
                to_remove.append(var)
                print(f"No observational and input data match for {var}, skipping")
        for var in to_remove:
            variables.remove(var)
        return varlist, variables

    def make_obs_comparisonplots(
        self, ilamb_cfgs, variables_obs_list=None, season="ANN", year_range_in=None
    ):
        if variables_obs_list is None:
            if self.var_pams["OBSERVATION_COMPARISON"] is None:
                print("This XesmfCLMFatesDiagnostics instance has no preprescribed observation comparison dataset")
                print("An explicit dictionary of variables and datasets to compare must be sent to get observational comparison plots")
                return
            variables_obs_list = self.var_pams["OBSERVATION_COMPARISON"]

        variables = list(variables_obs_list.keys())
        varlist, variables = self.make_alternate_varlist(ilamb_cfgs, variables)
        self.add_to_unit_dict(varlist)
        self.add_to_unit_dict(variables)
        if year_range_in is None:
            year_range = self.get_year_range()
        else:
            year_range = year_range_in
        if season == "ANN":

            outd = self.get_annual_data(year_range, varlist=varlist)
            season_name = season
        else:
            outd = self.get_seasonal_data(season, year_range, varlist=varlist)
            season_name = SEASONS[season]
        fig_dir = self.setup_folders_for_observation_plots(season_name)
        
        for var in variables:
            logscale = False
            if "LOG_PLOT" in self.var_pams and var in self.var_pams["LOG_PLOT"]:
                logscale = True
            ilamb_cfgs.print_var_dat(var)
            yminv, ymaxv, diffrange, negdiffrange = ilamb_cfgs.configurations[var].obs_limits
            varname_mod =  ilamb_cfgs.get_varname_in_file(var, self.var_pams["VAR_LIST_MAIN"])
            print(var)
            unit_conversion_factor, unit_to_print = get_unit_conversion_from_string(ilamb_cfgs.get_variable_plot_unit(var), self.unit_dict[varname_mod])
            print(f"{var}/{varname_mod} has unit conversion: {unit_conversion_factor} and new unit is {unit_to_print}")
            for obs_dataset in variables_obs_list[var]:
                to_plot_obs = ilamb_cfgs.get_data_for_map_plot(var, obs_dataset, self.regrid_target, season = season_name)
                
                if to_plot_obs is None:
                    print(f"Comparison for {var} to {obs_dataset} is currently unsupported for {season_name}")
                    continue
                fig, axs = plt.subplots(
                    nrows=1,
                    ncols=3,
                    figsize=(30, 10),
                    subplot_kw={"projection": ccrs.Robinson()},
                    layout = 'constrained'
                )
                to_plot = unit_conversion_factor * regrid_se_data(self.regridder, outd[varname_mod])
                

                year_range_str = f"{year_range[0]:04d}-{year_range[-1]:04d}"
                make_bias_plot(
                    to_plot,
                    f"{self.casename}",
                    ax=axs[0],
                    yminv = yminv,
                    ymaxv = ymaxv,
                    xlabel = f"{season} {varname_mod} [{unit_to_print}]",
                    logscale=logscale
                )
                make_bias_plot(
                    to_plot_obs,
                    f"{obs_dataset}",
                    ax=axs[1],
                    yminv = yminv,
                    ymaxv = ymaxv,
                    xlabel = f"{season} {varname_mod} [{unit_to_print}]",
                    logscale=logscale
                )
                make_bias_plot(
                    to_plot - to_plot_obs,
                    f"{self.casename} - {obs_dataset}",
                    yminv = negdiffrange,
                    ymaxv = diffrange,
                    ax=axs[2], 
                    cmap = "RdYlBu_r"
                )
                rmse = calculate_rmse_from_bias(to_plot - to_plot_obs)
                fig.suptitle(f"{season_name} {varname_mod} ({unit_to_print}) (years {year_range_str}), RMSE = {rmse}", size = "xx-large", y=0.8)
                fig.savefig(
                    f"{fig_dir}/{self.casename}_compare_{varname_mod}_{obs_dataset}_{season_name}_{year_range_str}.png"
                )