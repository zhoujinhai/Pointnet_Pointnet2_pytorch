import shutil
import os
import glob

bug_txt = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\bug.txt"
adjust_txt = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\adjust.txt"
pcd_dir = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd"
obj_dir = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\down_obj"
pts_dir = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pts"
save_dir = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\test"

bug_files = []
with open(bug_txt, "r") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        bug_files.append(line)

for bug_file in bug_files:
    basename = os.path.splitext(os.path.basename(bug_file))[0]
    pcd_file = os.path.join(obj_dir, basename + ".obj")
    shutil.move(pcd_file, os.path.join(save_dir, "obj"))

# adjusted_files = []

# with open(adjust_txt, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip()
#         if os.path.isfile(line):
#             adjusted_files.append(os.path.splitext(os.path.basename(line)))
#
# pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
