import numpy as np
import xarray as xr

UNIT_PREFIXES = {
    "d" : -1,
    "c" : -2,
    "m" : -3,
    "u" : -6,
    "n" : -9,
    "p" : -12,
    "f" : -15,
    "a" : -18,
    "h" : 2,
    "k" : 3,
    "M" : 6,
    "T" : 12,
    "P" : 15,
    "E" : 18
}

TIME_UNITS_IN_S = {
    "s" : 1,
    "h" : 3600,
    "d" : 3600*24,
    "y" : 365*3600*24
}

def simple_conversion_numbers(base_unit_in, base_unit_out):
    if base_unit_in in TIME_UNITS_IN_S and base_unit_out in TIME_UNITS_IN_S:
        return TIME_UNITS_IN_S[base_unit_in] / TIME_UNITS_IN_S [base_unit_out]
    print(f"Basic underlaying unit is not the same ({base_unit_to} vs {base_unit_from}), currently unimplemented")
    return 1

def do_light_unit_string_conversion(unit):
    if "/" in unit:
        unit_nom_denom = unit.split("/")
        unit_nom_denom[1] = unit_nom_denom[1].replace("^", "-")
        unit_nom_denom[0] = unit_nom_denom[0].replace("^", "")
        if not unit_nom_denom[1][-1].isdigit():
            unit_nom_denom[1] = f"{unit_nom_denom[1]}-1"
        unit = " ".join(unit_nom_denom)
    if "gC" in unit:
        unit = unit.replace("gC", "g")
    return unit



def get_unit_conversion_and_new_label(orig_unit):
    shift = 0
    if orig_unit == "K":
        shift = -273.15
        ylabel = "C"
    else:
        ylabel = orig_unit
    return shift, ylabel

def unit_convert_single_unit(unit_from, unit_to):
    factor = 1
    if unit_from == unit_to:
        return 1
    # TODO: This implementation assumes no exponent for nominator units
    if "-" in unit_to:
        factor = -int(unit_to.split("-")[-1])
        just_string_to = unit_to.split("-")[0]
        just_string_from = unit_from.split("-")[0]
    else:
        just_string_to = unit_to
        just_string_from = unit_from
    base_unit_to = just_string_to[-1]
    base_unit_from = just_string_from[-1]
    multiplicator = 1
    if base_unit_from != base_unit_to:
        multiplicator = simple_conversion_numbers(base_unit_from, base_unit_to)
    if len(just_string_from) > 1:
        from_prefix = UNIT_PREFIXES[just_string_from[0]]
    else:
        from_prefix = 0
    if len(just_string_to) > 1:
        to_prefix = UNIT_PREFIXES[just_string_to[0]]
    else:
        to_prefix = 0
    return (multiplicator*10**((from_prefix-to_prefix)))**factor


def get_unit_conversion_from_string(obs_unit, mod_unit):
    if obs_unit is None or mod_unit is None:
        print("Stopped on the None")
        return 1, mod_unit
    print(f"Coming in with obs: {obs_unit}  and mod: {mod_unit}")
    obs_unit_parts = obs_unit.split()
    mod_unit_parts = mod_unit.split()
    if len(obs_unit_parts) != len(mod_unit_parts):
        print("Units not directly compatible area weighting likely necessary, this is currently unimplemented")
        return 1, mod_unit
    unit_conversion = 1
    for partnum in range(len(obs_unit_parts)):
        unit_conversion = unit_conversion*unit_convert_single_unit(mod_unit_parts[partnum], obs_unit_parts[partnum])
    if unit_conversion == 1:
        return unit_conversion, mod_unit
    return unit_conversion, obs_unit




def make_regridding_target_from_weightfile(weight_file, filename_exmp):
    exmp_dataset = xr.open_dataset(filename_exmp)
    is_weight_file= True
    if "lon" in exmp_dataset.dims and "lat" in exmp_dataset.dims:
        is_weight_file = False
    if is_weight_file:
        weights = xr.open_dataset(weight_file)
        out_shape = weights.dst_grid_dims.load().data.tolist()[::-1]
        
        #Some prep to get the bounds:
        lat_b_out = np.zeros(out_shape[0]+1)
        lon_b_out = weights.xv_b.data[:out_shape[1]+1, 0]
        lat_b_out[:-1] = weights.yv_b.data[np.arange(out_shape[0])*out_shape[1],0]
        lat_b_out[-1] = weights.yv_b.data[-1,-1]
        dummy_out = xr.Dataset(
            {
                "lat": ("lat", weights.yc_b.data.reshape(out_shape)[:, 0]),
                "lon": ("lon", weights.xc_b.data.reshape(out_shape)[0, :]),
                #"lat_b": ("lat_b", lat_b_out),
                #"lon_b": ("lon_b", lon_b_out),
            }
        ) 
    else:
        dummy_out = xr.Dataset(
            {
                "lat": ("lat", exmp_dataset.lat.values),
                "lon": ("lon", exmp_dataset.lon.values),
                #"lat_b": ("lat_b", exmp_dataset.lat_b),
                #"lon_b": ("lon_b", exmp_dataset.lon_b),
            }
        )
    return dummy_out