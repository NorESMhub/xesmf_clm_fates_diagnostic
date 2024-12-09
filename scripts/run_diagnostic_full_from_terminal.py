import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from xesmf_clm_fates_diagnostic import XesmfCLMFatesDiagnostics

standard_run_dict = {
    "weight" : "/projects/NS9188K/NORESM_INTERIM_TEMP/map_files/map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc",
    "outpath" : "figs/",
    "pamfile" : f"{os.path.dirname(__file__)}/standard_pams.json",
    "compare": None
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
    print("If not supplied the file standard_pams.json will be used, feel free to copy that file as a template") # TODO: Make example parameterfile
    print(f"python {os.path.dirname(__file__)}/{os.path.basename(__file__)} --help will reiterate these instructions")
    sys.exit(4)

def read_optional_arguments(arguments):
    run_dict = standard_run_dict.copy()
    for arg in arguments:
        arg_key = arg.split("=")[0] 
        arg_val = arg.split("=")[-1] 
        if arg_key in run_dict:
            if not os.path.exists(arg_val):
                print(f"Invalid path {arg_val} for {arg_key} will be ignored")
            else:
                run_dict[arg_key] = arg_val
        else:
            print(f"Argument {arg} is not a valid argument and will be ignored")
        if not os.path.exists(run_dict["outpath"]):
            print(f"Output path {run_dict['outpath']} must exist")
            print_help_message()
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
if len(glob.glob(f"{run_path}/*.nc")) < 1:
    print("You must supply a path to land output data,  path to lnd/hist folder is expected")
    print(f"path {run_path} contains no netcdf files")
    print_help_message()

run_dict = read_optional_arguments(sys.argv[2:])
print(f"All set, setting up to run diagnostics on {run_path} using options:")
print(run_dict)

diagnostic = XesmfCLMFatesDiagnostics(
    # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
    # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
    run_path,
    run_dict["weight"],
    run_dict["pamfile"],
    outdir = run_dict["outpath"],
    region_def="/projects/NS9560K/diagnostics/noresm/packages/CLM_DIAG/code/resources/region_definitions.nc",
)

print("Standard diagnostics:")
#print(diagnostic.find_case_year_range())

diagnostic.make_all_plots_and_tables()

if run_dict["compare"] is None:
    print(f"Done, output should be in {run_dict['outpath']}")
else:
    print(f"Comparison diagnostics with {run_dict['compare']}")
    compare_variables = ["TSA", "RAIN", "SNOW", "FSR", "FSDS"]

    diasgnostic_other = XesmfCLMFatesDiagnostics(
        # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
        # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
        run_dict['compare'],
        run_dict["weight"],
        run_dict["pamfile"],
    )

    diagnostic.make_combined_changeplots(diasgnostic_other, compare_variables)
    print(f"Done, output should be in {run_dict['outpath']}")
