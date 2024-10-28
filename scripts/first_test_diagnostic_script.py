import os 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from xesmf_clm_fates_diagnostic import XesmfCLMFatesDiagnostics

diagnostic = XesmfCLMFatesDiagnostics(
    #"/cluster/projects/nn9560k/mvertens/cases/n1850.ne30_tn14.hybrid_fatessp.202401007",
    "/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/1850_fates_spinup/",
    "/projects/NS9188K/NORESM_INTERIM_TEMP/map_files/map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc"
    )

print(diagnostic.find_case_year_range())