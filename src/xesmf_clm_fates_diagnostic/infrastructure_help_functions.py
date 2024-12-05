import os, sys, glob
import json

def setup_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

def setup_nested_folder_structure_from_dict(root, subfolder_dict):
    print(root)
    print(subfolder_dict)
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
    with open(pam_file_path, "r") as jsonfile:
        data = json.load(jsonfile)
    print(data)
    



  
