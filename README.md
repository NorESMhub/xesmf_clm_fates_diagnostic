# xesmf_clm_fates_diagnostic

An XESMF based diagnostic tool to produce various plots and tables for CLM-FATES

Currently very much a work in progress

## Prerequisites

In order to use the tool you need to load and ESMF module and build an xesmf-containing conda environment on top of it. On Nird running, navigating to the folder called `scripts` and running:

```
. setup.sh
```
will automatically do this for you.

Otherwise you can install an environment using the file `conda_env.yaml` to install such an environment yourself using conda (or Miniforge). However, be aware that to do so you also need ESMF installed and loaded before building, and you need to loaded the same ESMF module when loading the environment. We recommend editing the `setup.sh` file to reflect the setup needed on your machine if you build your own environment this way.

## Usage

To run navigate to the folder called scripts (or extend paths for run-scripts to include the full path to that folder in the following commands) and run: 
```
. setup.sh
```

Then run 
```
python run_diagnostic_full_from_terminal.py path_1 weight=weight_path compare=opt_path_2 outpath=opt_out_path pamfile=pamfile_path
```
where `path_1` is the path to the lnd/hist folder containing your output.

The other arguments are optional:

`weight_path` is a path to a weight-file if the standard one is not to be used. For regular lat, lon runs, the weight-file is not used, and hence you can send a dumy argument, use the standard, or if not working on nird, the tool will try to add a dummy and run without if you don't send a weight_path.

`opt_path_2` is a path to output from a run you wish to compare to 

`outpath` is the path of where you want the output diagnostic figures filetree to go. If not sent the figures will be expected to go in a folder called figs situated in whatever directory you ran the command from.
If you want this to be viewable by web, choose a web-facing directory. For instance if you have access to the NS9560K account, make a subdirectory with the same name as your username in /datalake/NS9560K/www/diagnostics/noresm/ and make that your outpath, i.e. `outpath=/datalake/NS9560K/www/diagnostics/noresm/username`

`pamfile_path` is the path to a parameterfile in which you can specify which variables to plot in the various plots. 
This file should be a json-file containing the three (four) keyword arguments:
* VAR_LIST_MAIN - and this should be followed by the list of main variables to plot on maps and for trends
* SEASONAL_VARSETS - and this should be followed by a dictionary of named variable sets for which to plot seasonal cycles over the various regions
* COMPARE_VARIABLES - is optional, if included this should be followed by a list of variables to use to make comparison plots, if not sent but a comparison is still requested, the compare_variables will be taken from VAR_LIST_MAIN
* OBSERVATION_COMPARISON - is optional, if included this should be followed by a dictionaries with keys variables (with their ilamb-naming) followed by a list of ilamb datasets to which they should be compared. The specified variables must be present in the ilamb configuration file found in the tests/test-data folder, but you can expand that file with more variable specifications if you wish. The parameter file short_pams.json in the scripts folder includes an example of this section that will give comparison plots for lai, hfls, hfss and gpp.
If the no pamfile argument is sent, the file standard_pams.json  is used. Feel free to copy that file to use as a template when making your own parameterfile, but we recommend not editing the file itself.

With command line arguments, you can also control the comparison output
`compare_from_start` or `compare_from_end` allows you to set the comparison between the runs to consider the `n` first or last years of each of the comparison runs.
`compare_custom_year_range` gives you the option to choose specific years to compare on the format `yearstart-yearend` if you want the same range from both sets, 
or `yearstart-yearend_yearstart-yearend` if you want one range for the main dataset, and a different range for the second one.
`compare_seasonal=True` will add seasonal comparison (for both comparison to model output and observations) plots for the same year ranges.