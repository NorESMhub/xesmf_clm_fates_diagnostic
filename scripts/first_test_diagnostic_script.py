import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from xesmf_clm_fates_diagnostic import XesmfCLMFatesDiagnostics

standard_weight = "/projects/NS9188K/NORESM_INTERIM_TEMP/map_files/map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc"
diagnostic = XesmfCLMFatesDiagnostics(
    # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
    # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
    "/nird/datalake/NS9560K//noresm3/cases/n1850.ne30_tn14.hybrid_fatessp.202401007/lnd/hist/",
    standard_weight,
    region_def="/projects/NS9560K/diagnostics/noresm/packages/CLM_DIAG/code/resources/region_definitions.nc",
)

print(diagnostic.find_case_year_range())
#diagnostic.make_global_yearly_trends()
#sys.exit(4)
diagnostic.make_all_plots_and_tables()

comparison_files = "/nird/datalake/NS9560K/noresm3/cases/n1850.ne30_tn14.hybrid_clmbgc.202401004/lnd/hist/"
compare_variables = ["TSA", "RAIN", "SNOW", "FSR", "FSDS"]

diasgnostic_other = XesmfCLMFatesDiagnostics(
    # "/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
    # "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
    comparison_files,
    standard_weight,
    varlist=compare_variables,
)

diagnostic.make_combined_changeplots(diasgnostic_other, compare_variables)