def get_unit_conversion_and_new_label(orig_unit):
    shift = 0
    if orig_unit == "K":
        shift = -273.15
        ylabel = "C"
    else:
        ylabel = orig_unit
    return shift, ylabel