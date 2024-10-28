import numpy as np
#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xarray as xr

#import matplotlib.pyplot as plt

import os, sys
#import netCDF4 as nc4
import shutil
import warnings
warnings.filterwarnings('ignore')

import xesmf as xe
#import cartopy.crs as ccrs

import plotting_methods

gppfile='/projects/NS9188K/NORESM_INTERIM_TEMP/ILAMB_OBS/gpp_FLUXCOM.nc'
laifile='/projects/NS9188K/NORESM_INTERIM_TEMP/ILAMB_OBS/lai_0.5x0.5_MODIS.nc'

ds_gpp = xr.open_dataset(gppfile)
ds_lai = xr.open_dataset(laifile)

ds_gpp=ds_gpp.mean('time')
ds_lai=ds_lai.mean('time')
ds_lai['lat']=ds_lai.lat*-1

ds=ds_gpp

output_data ='/projects/NS9188K/NORESM_INTERIM_TEMP/temp_spinup_out/'
case_name = "1850_fates_spinup"
year = 80
outd = None
for month in range(12):
    mfile = f"{output_data}{case_name}/{case_name}.clm2.h0.{year:04d}-{month + 1:02d}.nc"
    outd_here = xr.open_dataset(mfile, engine='netcdf4')
    if not outd:
        outd = outd_here
    else: 
        outd = xr.concat([outd, outd_here], dim= 'time')

ds_out_gpp_ts=outd.FATES_GPP*outd.FATES_FRACTION
ds_out_lai_ts=outd.FATES_LAI*outd.FATES_FRACTION
ds_vegc_ts=outd.FATES_VEGC*outd.FATES_FRACTION
#ds_out_vegc_ts=[]

weight_file = "/projects/NS9188K/NORESM_INTERIM_TEMP/map_files/map_ne30pg3_to_0.5x0.5_nomask_aave_da_c180515.nc"

regridder = plotting_methods.make_se_regridder(weight_file=weight_file)

ds_out_gpp_ts = plotting_methods.regrid_se_data(regridder, ds_out_gpp_ts)
ds_out_lai_ts = plotting_methods.regrid_se_data(regridder, ds_out_lai_ts)
ds_out_vegc_ts = plotting_methods.regrid_se_data(regridder, ds_vegc_ts)

conv = 3600*24*1000
ds_out_gpp = np.multiply(ds_out_gpp_ts.mean('time'),conv)
ds_out_lai = ds_out_lai_ts.mean('time')
ds_out_vegc = ds_out_vegc_ts.mean('time')

plotting_methods.make_bias_plot(ds_out_gpp, f"gpp_output_{case_name}_{year}_absolute_map.png", 0, 10, cmap="viridis")
plotting_methods.make_bias_plot(ds_out_lai, f"lai_output_{case_name}_{year}_absolute_map.png", 0, 5, cmap="viridis")
plotting_methods.make_bias_plot(ds_out_vegc, f"vegc_output_{case_name}_{year}_absolute_map.png", 0, 15, cmap="viridis")

regridder_obsdata = xe.Regridder(ds, ds_out_gpp, 'bilinear', periodic=True)

gpp45deg = regridder_obsdata(ds_gpp['gpp'])
lai45deg = regridder_obsdata(ds_lai['lai'])
lai45deg['lat']=lai45deg.lat*-1

bias_gpp = ds_out_gpp - gpp45deg
bias_lai = ds_out_lai - lai45deg
plotting_methods.make_bias_plot(bias_gpp, f"gpp_output_{case_name}_{year}_bias_fluxcom_map.png", -3, 3)
plotting_methods.make_bias_plot(bias_lai, f"lai_output_{case_name}_{year}_bias_modis_map.png", -3, 3)