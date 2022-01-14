import vedo
import numpy as np
import glob
import os


def show_pcl_data(pcd_file):
    data = np.loadtxt(pcd_file, skiprows=10).astype(np.float32)
    points = data[:, :3]
    labels = data[:, 3]
    colours = ["grey", "blue", "red", "pink", "white", "brown", "yellow", "green", "black"]

    diff_label = np.unique(labels)
    # print(pcd_file, " res_label: ", diff_label)
    group_points = []
    for label in diff_label:
        point_group = points[labels == label]
        group_points.append(point_group)

    show_pts = []
    for i, point in enumerate(group_points):
        pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


if __name__ == "__main__":
    # pcd_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd\1.pcd"
    # show_pcl_data(pcd_path)
    pcd_dir = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd"
    pcd_files = glob.glob(os.path.join(pcd_dir, "*.pcd"))
    for i, pcd_path in enumerate(pcd_files):
        if 230 < i < 285:
            continue
        print("show {}th file: {}".format(i+1, pcd_path))
        show_pcl_data(pcd_path)