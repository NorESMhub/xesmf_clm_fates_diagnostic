import os
import sys
import glob

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from xesmf_clm_fates_diagnostic import XesmfCLMFatesDiagnostics, ilamb_configurations

standard_run_dict = {
    "weight" : "/datalake/NS9560K/diagnostics/land_xesmf_diag_data/map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc",
    "outpath" : "/datalake/NS9560K/www/diagnostics/noresm/zofias/",
    "pamfile" : f"{os.path.dirname(__file__)}/firediags_standard.json",
    "compare": None,
    "year_range_compare": None, 
}

run_dict_optional_arguments = {
    "compare_seasonal": False,
    "mute_trend": False,
    "mute_maps": False  
}

def print_help_message():
    print("Usage: ")
    print(f"python {os.path.dirname(__file__)}/{os.path.basename(__file__)} path_1 weight=weight_path compare=opt_path_2 outpath=opt_out_path opt_file=opt_file_path")
    print("path_1 is  non-optional, and should give the path to the lnd/hist folder of data to be plotted")
    print("All other arguments are optional, and are read using keywords")
    print("Arguments beyond the first one with incorrect keywords will simply be ignored")
    print("Optional arguments are:")
    print("weight=weight_path")
    print("weight_path should be a path to a weight file, otherwise the standard land ne30pg3_to_0.5x0.5 file will be used")
    print("compare=opt_path_2")
    print("Optional path to run to compare to should point to lnd/hist folder")
    print("outpath=opt_out_path")
    print("Path to where to put outputted diagnostic figures. If not supplied, folder figs under working directory will be assumed")
    print("pamfile=pamfile_path")
    print("Path to a json file with parameters such as which variables to plot in the various sets") 
    print("If not supplied the file standard_pams.json will be used, feel free to copy that file as a template")
    print("compare_from_start=number_of_years_from_start_to_compare")
    print("compare_from_end=number_of_years_from_end_to_compare")
    print("compare_custom_year_range=year-year_year-year")
    print("These three optional arguments allow you to specify the year ranges for comparison plots.")
    print("If you supply more than one of them, the last one supplied will be used")
    print("The custom year range can either be the same for both simulations, in which case year-year is sufficient")
    print("If the years in the two comparison sets are to be different, you need to supply year-year_year-year")
    print("where the first year-year denotes the range for the main dataset, and the second that of the comparison dataset")
    print("compare_seasonal=True")
    print("Adds seasonal comparison plots for both observations and comparison to other model output")
    print(" if comparison plotting and/ or observational comparisons are included. Yearranges for this will be the same as for annual comparison plots")
    print("Boolean keywords mute_trend=True or mute_maps=True can be added to mute the generation of total")
    print("timeseries trends or variable annual or seasonal single maps, this will")
    print("shave time off the runtime of the diagnostic if they are not of interest to you")
    print(f"python {os.path.dirname(__file__)}/{os.path.basename(__file__)} --help will reiterate these instructions")
    sys.exit(4)

def read_optional_arguments(arguments):
    run_dict = standard_run_dict.copy()
    print(arguments)
    for arg in arguments:
        print(arg)
        arg_key = arg.split("=")[0] 
        arg_val = arg.split("=")[-1] 
        print(arg_key)
        print(arg_val)
        if arg_key in run_dict:
            if not os.path.exists(arg_val):
                print(f"Invalid path {arg_val} for {arg_key} will be ignored")
            else:
                if os.path.isdir(arg_val) and arg_val[-1] != "/":
                    run_dict[arg_key] = f"{arg_val}/"
                else:
                    run_dict[arg_key] = arg_val
        elif arg_key in  ["compare_from_start", "compare_from_end"]:
            run_dict["year_range_compare"] = {arg_key:int(arg_val)}
        elif arg_key == "compare_custom_year_range":
            split = arg_val.split("_")
            run_dict["year_range_compare"] = {"year_range":np.arange(int(split[0].split("-")[0]), int(split[0].split("-")[-1]) +1)}
            if len(split) > 1:
                run_dict["year_range_compare"]["year_range_other"] = np.arange(int(split[1].split("-")[0]), int(split[1].split("-")[-1]) +1)
        elif arg_key in run_dict_optional_arguments:
            run_dict[arg_key] = bool(arg_val)
        elif arg_key in ["mute_trend", "mute_maps"]:
            run_dict[arg_key] = True
        else:
            print(f"Argument {arg} is not a valid argument and will be ignored")
        if not os.path.exists(run_dict["outpath"]):
            print(f"Output path {run_dict['outpath']} must exist")
            print_help_message()
    for arg_key_opt, arg_val_opt in run_dict_optional_arguments.items():
        if arg_key_opt not in run_dict:
            run_dict[arg_key_opt] = arg_val_opt

    # In case you are not working on NIRD, and forget to send weight-file
    if not os.path.exists(run_dict["weight"]):
        print("Weight file path does not exist. will run with dummy argument-file.")
        print("This will only work for regular lat-lon data")
        run_dict["weight"] = run_dict["pamfile"]
    return run_dict

# Making sure there is a run_path argument
if len(sys.argv) < 2:
    print("You must supply a path to land output data, path to lnd/hist folder is expected!")
    print_help_message()
if sys.argv[1] == "--help":
    print_help_message()
run_path = sys.argv[1]
if not os.path.exists(run_path):
    print("You must supply a path to land output data,  path to lnd/hist folder is expected!")
    print(f"path {run_path} does not exist")
    print_help_message()
if run_path[-1] != "/":
    run_path = f"{run_path}/"
if len(glob.glob(f"{run_path}*.nc")) < 1:
    print("You must supply a path to land output data,  path to lnd/hist folder is expected")
    print(f"path {run_path} contains no netcdf files")
    print_help_message()

ilamb_cfg = ilamb_configurations.IlambConfigurations("../tests/test-data/ilamb_CLMFATES.cfg")
#print(ilamb_cfg.configurations["FATES_VEGC"].obsdatasets)
#print(ilamb_cfg.configurations["pr"].obsdatasets)

#sys.exit(4)

run_dict = read_optional_arguments(sys.argv[2:])
#sys.exit(4)

print(f"All set, setting up to run diagnostics on {run_path} using options:")
print(run_dict)
#sys.exit(4)

diagnostic = XesmfCLMFatesDiagnostics(
    # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
    # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
    run_path,
    run_dict["weight"],
    run_dict["pamfile"],
    outdir = run_dict["outpath"],
    region_def="region_def_improved.nc",
)

print("Standard diagnostics:")
print(diagnostic.find_case_year_range())


#sys.exit(4)
diagnostic.make_all_plots_and_tables(ilamb_cfgs = ilamb_cfg, mute_trend=run_dict["mute_trend"], mute_maps=run_dict["mute_maps"])
#sys.exit(4)

if not run_dict["compare"] is None:
    print(f"Comparison diagnostics with {run_dict['compare']}")

    diasgnostic_other = XesmfCLMFatesDiagnostics(
        # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
        # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
        run_dict['compare'],
        run_dict["weight"],
        run_dict["pamfile"],
    )

    diagnostic.make_combined_changeplots(diasgnostic_other, year_range_in=run_dict["year_range_compare"], ilamb_cfgs = ilamb_cfg)
    if run_dict["compare_seasonal"]:
        print("Seasons true")
        for season in range(4):
            print(f"Comparison statistics with {season}")
            diagnostic.make_combined_changeplots(diasgnostic_other, season=season, year_range_in=run_dict["year_range_compare"], ilamb_cfgs = ilamb_cfg)

if diagnostic.var_pams["OBSERVATION_COMPARISON"] is not None:
    print("Doing observational comparisons")
    diagnostic.make_obs_comparisonplots(ilamb_cfg)
    if run_dict["compare_seasonal"]:
        for season in range(4):
            print(f"Seasonal observational comparisons with {season}")
            diagnostic.make_obs_comparisonplots(ilamb_cfg, season=season)

print(f"Done, output should be in {run_dict['outpath']}")
