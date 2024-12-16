#!/bin/bash
conda deactivate
source deactivate
module purge
module load Anaconda3/2022.05
module load ESMF/8.4.1-intel-2022a
conda activate /projects/NS9560K/diagnostics/land_xesmf_env/diag_xesmf_env/
