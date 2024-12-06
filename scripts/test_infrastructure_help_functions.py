import os
import sys
import glob

sys.path.append(os.path.join(os.path.dirname(__file__), "../", "src"))

from xesmf_clm_fates_diagnostic import infrastructure_help_functions

test_folder_setup = False
test_read_pamfile = True
if test_folder_setup: 
    here_t = f"{os.path.dirname(__file__)}/test"
    delete = True
    print(here_t)
    infrastructure_help_functions.setup_folder_if_not_exists(here_t)

    subfolder_dict = {
        "sub_1":{
            "sub_1_1": ["sub_1_1_1"],
            "sub_1_2": "sub_1_2_1"
        },
        "sub_2": None,
        "sub_3": {
            "sub_3_2": {"sub_3_2_1": "test"}      
        }
    }

    infrastructure_help_functions.setup_nested_folder_structure_from_dict(here_t, subfolder_dict)

    if os.path.exists("test") and delete:
        print("Deleting")
        infrastructure_help_functions.clean_empty_folders_in_tree(here_t)

if test_read_pamfile:
    data = infrastructure_help_functions.read_pam_file("standard_pams.json")
    print(data)
    print(type(data))
