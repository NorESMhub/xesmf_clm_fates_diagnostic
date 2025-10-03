import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import math
import xesmf
from matplotlib.colors import LogNorm

from  .misc_help_functions import get_unit_conversion_and_new_label

def make_bias_plot(bias,figname,yminv=None,ymaxv=None,cmap = 'viridis',ax = None, xlabel=None, logscale=False):
    # Use viridis for absolute maps

    print("in make bias plot",figname,bias.name)
    print_to_file = False
    if ax is None:
        print_to_file = True
    else:
        shrink = 0.5

    if ax is None:  # this is a single plot situation

        dims = list(bias.dims)
        if(len(dims) == 3): # we have an extra dimension
            # Find the "extra" dim (not lat/lon)
            extra_dim = [d for d in dims if d not in ["lat", "lon"]][0]
            n = bias.sizes[extra_dim]
            print("n",n)
            print(bias[extra_dim])

            # Try to get human-readable labels (if available)
            if n>0: #extra_dim in bias.coords:
#                labels = [str(l) for l in bias[extra_dim].values]
#            else:
                labels = [f"{extra_dim}={i}" for i in range(n)]
                ncols = math.ceil(math.sqrt(n))
                nrows = math.ceil(n / ncols)
                print("making 3d fig")
                fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), constrained_layout=True)
                images = []
                for i in range(n):
                    print("plotting pfts",i)
                    ax = axes[i // ncols, i % ncols]
                    im =bias.isel({extra_dim: i}).plot(ax=ax,vmin=yminv,vmax=ymaxv,add_colorbar=False)
                    ax.set_title(labels[i])

                    ax.set_xlabel('')
                    ax.set_xticks([])
                    ax.set_ylabel('')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    images.append(im)    

                #fig.colorbar(im, ax=axes, location='right', label=bias.name)
                    
                # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                # fig.colorbar(im, cax=cbar_ax, label=bias.name)

                # Hide unused axes
                for j in range(i+1, nrows*ncols):
                    fig.delaxes(axes[j // ncols, j % ncols])

                fig.suptitle(f"{bias.name}", fontsize=16)
                plt.tight_layout()

        else: # normal 2D plot.     
            fig = plt.figure(figsize=(10, 5))
            # Create a GeoAxes with the PlateCarree projection
            #ax = plt.axes(projection=ccrs.PlateCarree())
        
            ax = plt.axes(projection=ccrs.Robinson())
            print_to_file = True
            shrink = 0.7
    #print(figname)
    # Plot the data on the map

            if xlabel is not None:
                shift, xlabel = get_unit_conversion_and_new_label(xlabel.split("[")[-1][:-1])
                bias = bias + shift
            try:
                if (yminv is None) or (ymaxv is None):
                    if not logscale:
                        im = bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap)
                    else:
                        bias = bias.where(bias > 0)
                        im = bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap, norm = LogNorm())
                else:
                    im = bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap, vmin=yminv, vmax=ymaxv)
                cb =  im.colorbar
                cb.remove()
                plt.colorbar(im, ax=ax, shrink=shrink)#fraction=0.046, pad=0.04
            except TypeError as err:
                print(f"Not able to produce plot due to {err}")
                ax.clear()
            
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
    
    # Show the plot
    if print_to_file:
        fignamefull=figname+'.png'
        fig.savefig(fignamefull,bbox_inches='tight')

def make_bias_plot_latixy_longxy(bias,latixy, longxy, figname,yminv,ymaxv,cmap = 'RdYlBu_r', log_plot=False):
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
    #print(regrid_start)
    lat_min = np.argmin(np.abs((regrid_target["lat"].values - regrid_start["lat"].values.min())))
    lat_max = np.argmin(np.abs(regrid_target["lat"].values - regrid_start["lat"].values.max()))
    regrid_target = regrid_target.isel(lat=slice(lat_min, lat_max))
    #print(f"lat_min {lat_min}, lat_max: {lat_max}")# lon_min: {lon_min}, lon_max: {lon_max}")

    #print(regrid_target)
    return xesmf.Regridder(
        regrid_start,
        regrid_target,
        method = method,
        periodic = True,
        #reuse_weights=True
    )

def make_regridder_regular_to_coarsest_resolution(regrid_target1, regrid_target2):
    if (regrid_target2.lat.shape[0] == regrid_target1.lat.shape[0]) and (regrid_target2.lon.shape[0] == regrid_target1.lon.shape[0]):
        return None, False
    if regrid_target1.lat.shape[0] > regrid_target2.lat.shape[0]:
        regridder_here = make_regular_grid_regridder(regrid_target1, regrid_target2)
        return regridder_here, True
    regridder_here = make_regular_grid_regridder(regrid_target2, regrid_target1)
    return regridder_here, False


