
import os

import numpy as np
import xarray as xr

from .plotting_methods import make_regular_grid_regridder
from .misc_help_functions import get_unit_conversion_from_string, get_unit_conversion_and_new_label, SEASONS

class IlambCompVariable:

    def __init__(self, name):
        self.name = name
        self.alt_names = None
        self.obsdatasets = None
        self.plot_unit = None
        self.obs_limits = [None, None, None, None]
    
    def set_alt_names(self, alt_name_string):
        alt_names = []
        for alt_name in alt_name_string.split(","):
            alt_names.append(alt_name)
        self.alt_names = alt_names

    def add_obsdataset(self, path_string):
        if self.obsdatasets is None:
            self.obsdatasets = {}
        oname = path_string.split("/")[-2]
        self.obsdatasets[oname] = {"dataloc": path_string, "conv_factor": 1}
        return oname
    
    def calc_obs_conv_factor(self, oname, plot_unit, tab_unit):
        if plot_unit != tab_unit:
            unit_conversion, obs_unit =  get_unit_conversion_from_string(plot_unit, tab_unit)
            #print(f"For {oname} with table_unit {tab_unit} and plot_unit {plot_unit} we get conversion factor {unit_conversion} to get obs_unit {obs_unit}")
            self.obsdatasets[oname]["conv_factor"] = unit_conversion
    
    def set_plot_unit(self, plot_unit, oname):
        if self.plot_unit is None:
            self.plot_unit = plot_unit
        else:
            tab_unit = self.plot_unit
            self.plot_unit = plot_unit
            self.calc_obs_conv_factor(oname, plot_unit, tab_unit)
        

    def set_obs_limits(self, lim_string):
        obs_limits = np.array(lim_string.split(",")).astype(float)
        self.obs_limits = np.append(obs_limits, -obs_limits[-1])
        

def read_ilamb_configurations(cfg_file):
    ilamb_cfgs = {}
    curr_var = None
    curr_oname = None
    with open(cfg_file, "r") as cfile:
        for line in cfile:
            #print(line)
            if line.startswith("variable"):
                if curr_var is not None:
                    ilamb_cfgs[curr_var.name] = curr_var
                    curr_oname = None
                curr_var = IlambCompVariable(line.split('"')[-2].strip())
                #print(curr_var.name)
            if line.startswith("alternate_vars"):
                curr_var.set_alt_names(line.split('"')[-2].strip())
            if line.startswith("limits"):
                curr_var.set_obs_limits(line.split("=")[-1].strip())
            if line.startswith("source"):
                curr_oname = curr_var.add_obsdataset(line.split('"')[-2].strip())
            if line.startswith("table_unit"):
                tab_unit = line.split('"')[-2].strip()
                #print(f"Now processing {tab_unit} for {curr_oname} and {curr_var.name} that has ")
                if curr_var.plot_unit is not None and curr_var.plot_unit != tab_unit:
                    curr_var.calc_obs_conv_factor(curr_oname, plot_unit, tab_unit)
                elif curr_var.plot_unit is None:
                    curr_var.set_plot_unit(tab_unit, curr_oname)         
            if line.startswith("plot_unit"):
                plot_unit = line.split('"')[-2].strip()
                curr_var.set_plot_unit(plot_unit, curr_oname)
    if curr_var is not None:
        ilamb_cfgs[curr_var.name] = curr_var
    return ilamb_cfgs


class IlambConfigurations:

    def __init__(self, cfg_file, ilamb_data_dir="/datalake/NS9560K/diagnostics/ILAMB-Data/"):
        self.data_root = ilamb_data_dir
        if isinstance(cfg_file, dict):
            self.configurations = cfg_file
        elif os.path.exists(cfg_file):
            self.configurations = read_ilamb_configurations(cfg_file)
        else:
            self.configurations = None


    def get_filepath(self, variable, oname):
        return os.path.join(self.data_root, self.configurations[variable].obsdatasets[oname]["dataloc"])
    
    def get_varname_in_file(self, variable, dataset_keys):
        if variable in dataset_keys:
            return variable
        if variable in self.configurations:
            for alt_name in self.configurations[variable].alt_names:
                if alt_name in dataset_keys:
                    return alt_name
        return None
    
    def get_vaname_in_ilamb_cfgs(self, variable):
        if variable in self.configurations:
            return variable
        for ilambvar, cfgs in self.configurations.items():
            if cfgs.alt_names is None:
                continue
            if variable in cfgs.alt_names:
                return ilambvar
        return None
        
    def get_monthly_mean_timeslice_dataset_for_variable_obs(self, variable, oname, year_range=None, season="ANN"):
        dataset = xr.open_dataset(self.get_filepath(variable, oname))
        time_len = len(dataset["time"])
        varname = self.get_varname_in_file(variable, dataset.keys())
        if time_len%12 != 0:
            return None
      
        #if year_range is None:
            #year_range = range(np.max(time_len-120, 0), time_len)
        start_index = int(np.max(time_len-240, 0))
        outd_gn = dataset[varname].isel(time = slice(start_index, time_len))
        if "missing" in dataset[varname].attrs.keys():
            print(f"{varname} has missing with value {dataset[varname].attrs["missing"]}")
        outd_gn = outd_gn.where(outd_gn < 1e9)

        monthly_means = outd_gn.groupby('time.month').mean('time')
        print(f"Mean value of {monthly_means.mean().values} and conversion factor {self.configurations[variable].obsdatasets[oname]['conv_factor']}")
        return monthly_means * self.configurations[variable].obsdatasets[oname]["conv_factor"]

    
    def get_data_for_map_plot(self, variable, oname, regrid_target, season="ANN", year_range = None, clim_length = 20):
        #path = self.get_filepath(variable, oname)       
        # TODO: Deal with year_range not None
        if year_range is not None:
            year_range = None
        if not os.path.exists(self.get_filepath(variable, oname)):
            print(f"Observation in path {self.get_filepath(variable, oname)} not found, check your configuration files")
            return None
        
        dataset = xr.open_dataset(self.get_filepath(variable, oname))
        time_len = len(dataset["time"])
        varname = self.get_varname_in_file(variable, dataset.keys())
        if time_len%12 != 0 and season != "ANN":
            return None
        if time_len%12 == 0:
            #if year_range is None:
                #year_range = range(np.max(time_len-120, 0), time_len)
            clim_length = int(np.min((clim_length, time_len//12)))
            start_index = int(np.max((time_len-clim_length*12, 0)))
            if season == "ANN":
                t_pick = slice(start_index, time_len)
            elif season in SEASONS:
                idx = SEASONS.index(season)
                in_year_idx = np.mod(idx*3 - 1 + np.arange(3), 12)
                t_pick = np.array([start_index + yr*12 + mnth for yr in np.arange(clim_length) for mnth in in_year_idx])
                t_pick.sort()
            else:
                pass
            #print(t_pick)
            outd_gn = dataset[varname].isel(time = t_pick).mean(dim="time")
            
        elif year_range is None:
            start_index = np.max(time_len-clim_length, 0) -1
            outd_gn = dataset[varname].isel(time = slice(start_index, time_len)).mean(dim="time")
        if "missing" in dataset[varname].attrs.keys():
            print(f"{varname} has missing with value {dataset[varname].attrs["missing"]}")
        outd_gn = outd_gn.where(outd_gn < 1e9)
        regridder = make_regular_grid_regridder(outd_gn, regrid_target)
        output = regridder(outd_gn)
        regridder.grid_in.destroy()
        regridder.grid_out.destroy()
        del regridder
        #print(f"Dataset {oname} has conversion factor {self.configurations[variable].obsdatasets[oname]["conv_factor"]} for variable {variable}")
        return output * self.configurations[variable].obsdatasets[oname]["conv_factor"]
        
    def get_variable_plot_unit(self, variable):
        #print(variable)
        if variable in self.configurations.keys():
            return self.configurations[variable].plot_unit
        for ilamb_varname, ilamb_var in self.configurations.items():
            if ilamb_var.alt_names is None:
                continue
            if variable in ilamb_var.alt_names:
                return self.configurations[ilamb_varname].plot_unit
            
    
    def add_seasonal_obsdata_to_axis(self, figs, varlist, region_df, obs_comp_dict):
        rownum = int(np.ceil(len(varlist) / 2))
        print(obs_comp_dict)
        #sys.exit(4)
        for varnum, variable in enumerate(varlist):
            #varname = self.get_varname_in_file(variable)
            if variable is None:
                continue
            altname = self.get_vaname_in_ilamb_cfgs(variable)
            if altname is None:
                continue
            if altname not in obs_comp_dict:
                continue
            yminv, ymaxv, diffrange, negdiffrange = self.configurations[altname].obs_limits
            print(altname)
            for oname in obs_comp_dict[altname]:
                print(oname)
                outd = self.get_monthly_mean_timeslice_dataset_for_variable_obs(altname, oname)
                if altname == "TSA" and outd.mean()> 200:
                    outd = outd - 273.15
                #shift, ylabel = get_unit_conversion_and_new_label(outd.attrs["units"])
                #print(f"{altname} and shift: {shift}")
                #if oname == "FLUXCOM":
                #    print(f"{outd.lat.values}")
                #    print(f"{outd.lon.values}")
                for region, region_info in region_df.iterrows():
                    #print(f"{region}: {region_info}")
                    if rownum < 2:
                        axnow = figs[region][1][varnum % 2]
                    else: 
                        axnow = figs[region][1][varnum // 2, varnum % 2]
                    crop = outd.sel(
                        lat=(outd.lat >= region_info["BOX_S"]) & (outd.lat <= region_info["BOX_N"])
                        #lon=slice(region_info["BOX_W"], region_info["BOX_E"]),
                    )
                    if region_info["BOX_W"]%360 > region_info["BOX_E"]%360:
                        crop = crop.sel(lon=(crop.lon%360 >= region_info["BOX_W"]%360) | (crop.lon%360 <=region_info["BOX_E"]%360))
                    elif region_info["BOX_W"]%360 == region_info["BOX_E"]%360:
                        crop = crop
                    else:
                        crop = crop.sel(lon=(crop.lon%360 >= region_info["BOX_W"]%360) & (crop.lon%360 <=region_info["BOX_E"]%360))
                    #print(region_info)
                    #if region_info['PTITSTR'].strip() == 'Global' and oname == "FLUXCOM":
                    #    print(crop.lat.values)
                    #    print(crop.lon.values)
                    weights = np.cos(np.deg2rad(crop.lat))
                    weighted_data = crop.weighted(weights)
                    ts_data = weighted_data.mean(["lon", "lat"])
                    axnow.plot(range(12), ts_data, label=oname, ls='--')
                    axnow.set_title(f"{variable} vs {', '.join(obs_comp_dict[altname])}")
                    if yminv is not None and "pr" in self.configurations[altname].alt_names:
                        axnow.set_ylim(yminv, ymaxv)
            

    
    def print_var_dat(self, variable):
        print(self.configurations.keys())
        print(f"{self.configurations[variable].name} has alt_names: {self.configurations[variable].alt_names}, and plot unit: {self.configurations[variable].plot_unit}")
        

