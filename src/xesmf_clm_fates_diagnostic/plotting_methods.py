import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import xarray as xr
import xesmf


def make_bias_plot(bias,figname,yminv=None,ymaxv=None,cmap = 'RdYlBu_r',ax = None):
    # Use viridis for absolute maps
    print_to_file = False
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        # Create a GeoAxes with the PlateCarree projection
        #ax = plt.axes(projection=ccrs.PlateCarree())
        
        ax = plt.axes(projection=ccrs.Robinson())
        print_to_file = True
    
    # Plot the data on the map
    if (yminv is None) or (ymaxv is None):
        bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap)
    else:
        bias.plot(ax=ax, transform=ccrs.PlateCarree(),cmap=cmap, vmin=yminv, vmax=ymaxv)        
    ax.set_title('')
    ax.set_title(figname.split("/")[-1])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.coastlines()
    
    # Show the plot
    if print_to_file:
        fignamefull=figname+'.png'
        plt.savefig(fignamefull,bbox_inches='tight')

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

    
def make_se_regridder(weight_file):
    weights = xr.open_dataset(weight_file)
    in_shape = weights.src_grid_dims.load().data

    # Since xESMF expects 2D vars, we'll insert a dummy dimension of size-1
    if len(in_shape) == 1:
        in_shape = [1, in_shape.item()]

    # output variable shape
    out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]

    #print(in_shape, out_shape)

    dummy_in = xr.Dataset(
        {
            "lat": ("lat", np.empty((in_shape[0],))),
            "lon": ("lon", np.empty((in_shape[1],))),
        }
    )
    dummy_out = xr.Dataset(
        {
            "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
            "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
        }
    )

    regridder = xesmf.Regridder(
        dummy_in,
        dummy_out,
        weights=weight_file,
        method="bilinear",
        reuse_weights=True,
        periodic=True,
    )
    return regridder

def regrid_se_data(regridder, data_to_regrid):
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

