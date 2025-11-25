import os, sys, glob, shutil


def move_file_from_folder_ens_member(
        root_target = "/datalake/NS9560K/www/diagnostics/noresm/masan/PPE/ppe_runs_landonly", 
        root_orig = "figs/ppe",
        ens_member = 0,
        subfolder_name = "netcdf_dumps",
        copy = True
    ):
    ens_member_name = f"ensemble_member.{ens_member:03d}"
    for file in glob.glob(f"{root_orig}/{ens_member_name}/{subfolder_name}/*"):
        fname = file.split("/")[-1]
        target_path = f"{root_target}/{ens_member_name}/{subfolder_name}/{fname}"
        #print(f"Now moving {file} to {target_path}")
        if copy:
            shutil.copy(file, target_path)
        else: 
            shutil.move(file, target_path)

ensmembers = [ 0, 3, 5, 9, 19, 21, 24, 30, 32, 39, 41, 42, 48, 49, 58, 59, 64, 65, 69, 72, 73]
seasons = ["ANN", "DJF", "MAM", "JJA", "SON"]
for ens_member in ensmembers:
    for season in seasons:
        move_file_from_folder_ens_member(ens_member=ens_member, subfolder_name=f"OBS_comparison/{season}")