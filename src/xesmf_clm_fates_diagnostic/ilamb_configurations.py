
import os

import numpy as np
import xarray as xr

from .plotting_methods import make_regular_grid_regridder, regrid_se_data

class IlambCompVariable:

    def __init__(self, name):
        self.name = name
        self.alt_names = None
        self.obsdatasets = None
        self.plot_unit = None
    
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
        self.obsdatasets[oname]["conv_factor"] = "Unimplemented"
    
    def set_plot_unit(self, plot_unit):
        self.plot_unit = plot_unit
        

def read_ilamb_configurations(cfg_file):
    ilamb_cfgs = {}
    curr_var = None
    curr_oname = None
    curr_tab_unit = None
    with open(cfg_file, "r") as cfile:
        for line in cfile:
            #print(line)
            if line.startswith("variable"):
                if curr_var is not None:
                    ilamb_cfgs[curr_var.name] = curr_var
                    curr_oname = None
                    curr_tab_unit = None
                curr_var = IlambCompVariable(line.split('"')[-2].strip())
            if line.startswith("alternate_vars"):
                curr_var.set_alt_names(line.split('"')[-2].strip())
            if line.startswith("source"):
                curr_oname = curr_var.add_obsdataset(line.split('"')[-2].strip())
            if line.startswith("table_unit"):
                curr_tab_unit = line.split('"')[-2].strip()
            if line.startswith("plot_unit"):
                plot_unit = line.split('"')[-2].strip()
                if plot_unit != curr_tab_unit:
                    curr_var.calc_obs_conv_factor(curr_oname, plot_unit, curr_tab_unit)
                curr_var.set_plot_unit(plot_unit)
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
        

    
    def get_data_for_map_plot(self, variable, oname, regrid_target, season="ANN", year_range = None):
        #path = self.get_filepath(variable, oname)
        # TODO: Implement seasonal
        if season != "ANN":
            return None
        
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
        if time_len%12 == 1:
            #if year_range is None:
                #year_range = range(np.max(time_len-120, 0), time_len)
            start_index = int(np.max(time_len-120, 0)) -1
            outd_gn = dataset[varname].isel(time = slice(start_index, time_len)).mean(dim="time")
            
        elif year_range is None:
            start_index = np.max(time_len-10, 0) -1
            outd_gn = dataset[varname].isel(time = slice(start_index, time_len)).mean(dim="time")
        regridder = make_regular_grid_regridder(dataset, regrid_target)

        return regridder(outd_gn)
        
    def get_variable_plot_unit(self, variable):
        return self.configurations[variable].plot_unit
    
    def print_var_dat(self, variable):
        print(self.configurations.keys())
        print(f"{self.configurations[variable].name} has alt_names: {self.configurations[variable].alt_names}, and plot unit: {self.configurations[variable].plot_unit}")
        

