"""
descr: 显示txt文件或者pcd文件格式下的点云，不同标签数据予以不同颜色
Author: zjh
Data: 2021/10/19

"""
import os
import numpy as np
import vedo


def load_txt(txt_path):
    data = np.loadtxt(txt_path).astype(np.float32)
    labels = data[:, -1].astype(np.int32)
    diff_label = np.unique(labels)
    group_points = []
    for label in diff_label:
        point_group = data[labels == label]
        group_points.append(point_group[:, :3])
    return np.asarray(group_points, dtype=object)


def load_pcd(pcd_path, label_cls=-2):
    data = np.loadtxt(pcd_path, skiprows=10)   # skip head lines
    labels = data[:, label_cls].astype(np.int32)
    diff_label = np.unique(labels)
    group_points = []
    for label in diff_label:
        point_group = data[labels == label]
        group_points.append(point_group[:, :3])
    return np.asarray(group_points, dtype=object)


def show_pcl(file_path):
    # 根据最后一个标签分组显示
    ext = os.path.splitext(file_path)[-1]
    colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]
    points = None
    if ext == ".txt":
        points = load_txt(file_path)
    elif ext == ".pcd":
        points = load_pcd(file_path)
    else:
        print("Unknown file type! Just support txt or pcd file!")

    if points is None:
        print("file has not points! Please check file: {}".format(file_path))
    else:
        show_pts = []
        for i, point in enumerate(points):
            pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
            show_pts.append(pt)
        vedo.show(show_pts)


def show_pcl_data(data, label_cls=-1):
    colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]
    labels = data[:, label_cls].astype(np.int32)  # 最后一列为标签列
    diff_label = np.unique(labels)
    group_points = []
    for label in diff_label:
        point_group = data[labels == label]
        group_points.append(point_group[:, :3])
    points = np.asarray(group_points, dtype=object)

    show_pts = []
    for i, point in enumerate(points):
        pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


if __name__ == "__main__":
    # file_path = r"/home/heygears/jinhai_zhou/data/shapenetcore_partanno_segmentation_benchmark_v0_normal/02691156/1021a0914a7207aff927ed529ad90a11.txt"
    # file_path = r"C:\Users\Administrator\sse-images\0824-fangshedaoban-kehushuju (2)_minCruv.pcd"
    # show_pcl(file_path)

    import glob
    files = glob.glob(os.path.join(r"D:\Debug_dir\pcd_with_label", "*.pcd"))
    for i, file in enumerate(files):

        print(i, file)
        show_pcl(file)

    # hard data
    # 0824-fangshedaoban-kehushuju (118)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (22)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (38)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (39)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (48)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (53)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (62)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (63)_minCruv.pcd
    # 0824-fangshedaoban-kehushuju (71)_minCruv.pcd
