import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf

from  .misc_help_functions import get_unit_conversion_and_new_label


def make_bias_plot(bias,figname,yminv=None,ymaxv=None,cmap = 'viridis',ax = None, xlabel=None):
    # Use viridis for absolute maps
    print_to_file = False
    if ax is None:
        print_to_file = True
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        # Create a GeoAxes with the PlateCarree projection
        #ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax = plt.axes(projection=ccrs.Robinson())
        print_to_file = True
        shrink = 0.7
    else:
        shrink = 0.5
    
    # Plot the data on the map
    if xlabel is not None:
        shift, xlabel = get_unit_conversion_and_new_label(xlabel.split("[")[-1][:-1])
        bias = bias + shift

                       

    if (yminv is None) or (ymaxv is None):
        im = bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap)
    else:
        im = bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap, vmin=yminv, vmax=ymaxv)
              
    
    ax.set_title('')
    ax.set_title(figname.split("/")[-1])

    if xlabel is None:
        ax.set_xlabel('')
    else:
        ax.set_xticks([])
        ax.set_xlabel(xlabel)
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.coastlines()
    cb =  im.colorbar
    cb.remove()
    #cb.
    plt.colorbar(im, ax=ax, shrink=shrink)#fraction=0.046, pad=0.04) 

    
    # Show the plot
    if print_to_file:
        fignamefull=figname+'.png'
        fig.savefig(fignamefull,bbox_inches='tight')

def make_bias_plot_latixy_longxy(bias,latixy, longxy, figname,yminv,ymaxv,cmap = 'RdYlBu_r'):
    # Use viridis for absolute maps
    fig = plt.figure(figsize=(10, 5))
    # Create a GeoAxes with the PlateCarree projection
    #ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax = plt.axes(projection=ccrs.Robinson())
    
    # Plot the data on the map
    filled_c = ax.contourf(longxy, latixy, bias, cmap=cmap, transform=ccrs.PlateCarree(), vmin=yminv, vmax=ymaxv)
    ax.set_title('')
    ax.set_title(figname.split("/")[-1])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.coastlines()
    fig.colorbar(filled_c, vmin=yminv, vmax=ymaxv)
    
    # Show the plot
    fignamefull=figname+'.png'
    plt.savefig(fignamefull,bbox_inches='tight')

def make_generic_regridder(weightfile, filename_exmp):
    exmp_dataset = xr.open_dataset(filename_exmp)
    if "lon" in exmp_dataset.dims and "lat" in exmp_dataset.dims:
        return None
    else:
        return make_se_regridder(weight_file=weightfile)
    
def make_se_regridder(weight_file, regrid_method="conserved"):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    #print(in_shape, out_shape)

    #Some prep to get the bounds:
    lat_b_out = np.zeros(out_shape[0]+1)
    lon_b_out = weights.xv_b.data[:out_shape[1]+1, 0]
    lat_b_out[:-1] = weights.yv_b.data[np.arange(out_shape[0])*out_shape[1],0]
    lat_b_out[-1] = weights.yv_b.data[-1,-1]

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
            "lat_b": ("lat_b", np.empty((in_shape[0] + 1,))),
            "lon_b": ("lon_b", np.empty((in_shape[1] + 1,))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
            "lat_b": ("lat_b", lat_b_out),
            "lon_b": ("lon_b", lon_b_out),
        }
    )

    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method=regrid_method,#"conservative_normed",
        #method="bilinear",
        reuse_weights=True,
        periodic=True,
    )
    return regridder

def regrid_se_data(regridder, data_to_regrid):
    if regridder is None:
        return data_to_regrid
    #print(data_to_regrid.dims)
    if isinstance(data_to_regrid, xr.DataArray):
        #print(type(data_to_regrid))
        updated = data_to_regrid.copy().transpose(..., "lndgrid").expand_dims("dummy", axis=-2)
    else:
        vars_with_ncol = [name for name in data_to_regrid.variables if "ncol" in data_to_regrid[name].dims]
        updated = data_to_regrid.copy().update(
            data_to_regrid[vars_with_ncol].transpose(..., "lndgrid").expand_dims("dummy", axis=-2)
        )
    regridded = regridder(updated.rename({"dummy": "lat", "lndgrid": "lon"}))
    return regridded

def make_regular_grid_regridder(regrid_start, regrid_target, method= "bilinear"):
    print(regrid_start)
    lat_min = np.argmin(np.abs((regrid_target["lat"].values - regrid_start["lat"].values.min())))
    lat_max = np.argmin(np.abs(regrid_target["lat"].values - regrid_start["lat"].values.max()))
    regrid_target = regrid_target.isel(lat=slice(lat_min, lat_max))
    print(f"lat_min {lat_min}, lat_max: {lat_max}")# lon_min: {lon_min}, lon_max: {lon_max}")

    print(regrid_target)
    return xesmf.Regridder(
        regrid_start,
        regrid_target,
        method = method,
        periodic = True,
        #reuse_weights=True
    )

