import os, sys, glob
import json

def setup_folder_if_not_exists(path):
    """
    Create a folder if it does not exist already

    Parameters
    ----------
    path : str
        Path to folder that should be created

    """
    # TODO: Throw error 
    if not os.path.exists(path):
        os.mkdir(path)

def setup_nested_folder_structure_from_dict(root, subfolder_dict):
    setup_folder_if_not_exists(root)  
    if isinstance(subfolder_dict, dict):
        for key, value in subfolder_dict.items():
            setup_nested_folder_structure_from_dict(f"{root}/{key}", value)
    elif isinstance(subfolder_dict, list):
        for value in subfolder_dict:
            setup_folder_if_not_exists(f"{root}/{value}")
    elif isinstance(subfolder_dict, str):
        setup_folder_if_not_exists(f"{root}/{subfolder_dict}")
    return

def clean_empty_folders_in_tree(root):
    empty_below = 0
    for sub in glob.glob(root):
        if os.path.isdir(sub):
            empty_below = empty_below + clean_empty_folders_in_tree(sub)
        else:
            empty_below = empty_below + 1
    if empty_below == 0:
        os.rmdir(root)
        return 0
    return empty_below

def read_pam_file(pam_file_path):
    required = {"VAR_LIST_MAIN": list, "SEASONAL_VARSETS":dict, "COMPARE_VARIABLES":list}
    try:
        with open(pam_file_path, "r") as jsonfile:
            data = json.load(jsonfile)
    except json.decoder.JSONDecodeError as err:
        print(err)
        print(f"{pam_file_path} must be a valid json-file")
        sys.exit(4)
    if not isinstance(data, dict):
        raise ValueError(f"{pam_file_path} must evaluate to dict")
    for elem, e_type in required.items():
        if elem not in data.keys():
            if (elem == "COMPARE_VARIABLES") and ("VAR_LIST_MAIN" in data.keys()):
                data["COMPARE_VARIABLES"] = data["VAR_LIST_MAIN"]
            else:
                raise ValueError(f"{pam_file_path} must include {elem}")
        if not isinstance(data[elem], e_type):
            raise TypeError(f"{pam_file_path} element {elem} must be a {e_type}, but is {type(data[elem])}")
    return data
    



  
