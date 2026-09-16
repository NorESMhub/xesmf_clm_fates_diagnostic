#!/bin/bash
module purge
module load ESMF/8.6.0-foss-2023a
module load Miniforge3/24.1.2-0
conda deactivate
conda activate /nird/datalake/NS16000B/xesmf-env/
