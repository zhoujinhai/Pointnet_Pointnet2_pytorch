import sys
sys.path.append("E:/code/algorithm2/distribution/python/Release")
import hgapi
import numpy as np
import os
import vedo
import time
import glob
from multiprocessing import Process
# from tqdm import tqdm
import shutil


def get_gum_line_pts(gum_line_path):
    """
    读取牙龈线文件，pts格式
    """
    f = open(gum_line_path)
    pts = []

    is_generate_by_streamflow = False  # 是否由前台界面生成

    for num, line in enumerate(f):
        if 0 == num and line.strip() == "BEGIN_0":
            is_generate_by_streamflow = True
            continue
        if line.strip() == "BEGIN" or line.strip() == "END":
            continue
        if is_generate_by_streamflow:
            line = line.strip()
            if line == "END_0":
                pts.append(pts[0])
            else:
                splitted_line = line.split()
                point = [float(i) for i in splitted_line][:3]  # 只取点坐标，去掉法向量
                assert len(point) == 3, "点的坐标为x,y,z"
                pts.append(point)
        else:
            line = line.strip()
            splitted_line = line.split()
            point = [float(i) for i in splitted_line]
            assert len(point) == 3, "点的坐标为x,y,z"
            pts.append(point)

    f.close()
    pts = np.asarray(pts)
    return pts


# 生成牙龈线 v3
def simplify(model_path, save_path):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    try:
        basename = os.path.basename(model_path)
        file_name = os.path.splitext(basename)[0]
        save_name = os.path.join(save_path, file_name + ".obj")

        # 读取模型
        mesh = hgapi.Mesh()
        stl_handler = hgapi.STLFileHandler()
        stl_handler.Read(model_path, mesh)

        mesh_healing = hgapi.MeshHealing()
        mesh_healing.RemoveSmallerIslands(mesh)
        mesh_healing.RemoveDanglingFace(mesh)
        hgapi.MeshSimplify.Simplify2(mesh, 10000)

        # 保存模型
        obj_handler = hgapi.OBJFileHandler()
        obj_handler.Write(save_name, mesh)

    except Exception as e:
        with open('./simplify_error_model.txt', mode='a') as f:
            f.write(str(os.path.basename(model_path)))
            # f.write(" ")
            # f.write(repr(e))
            f.write('\n')  # 换行


def show_stl_pts(stl_path, pts_path):
    stl_model = vedo.load(stl_path).c(("magenta"))
    pts_point = get_gum_line_pts(pts_path)
    point = vedo.Points(pts_point.reshape(-1, 3)).pointSize(10).c(("green"))
    vedo.show(stl_model, point)


def mesh_simplify(stl_model_list, save_path):
    """
    批量处理预测模型
    @predict_model_list: 预测的模型列表
    @predict_path: 预测模型存放路径
    return: None
    """
    # for i, predict_model in enumerate(tqdm(stl_model_list)):
    for i, predict_model in enumerate(stl_model_list):
        print(predict_model)
        simplify(predict_model, save_path)


def parallel_simplify(model_list, save_path, n_workers=8):
    """
    多进程处理
    """
    if len(model_list) < n_workers:
        n_workers = len(model_list)
    chunk_len = len(model_list) // n_workers

    chunk_lists = [model_list[i:i + chunk_len] for i in range(0, (n_workers - 1) * chunk_len, chunk_len)]
    chunk_lists.append(model_list[(n_workers - 1) * chunk_len:])

    process_list = [Process(target=mesh_simplify, args=(chunk_list, save_path,)) for chunk_list in chunk_lists]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()


if __name__ == "__main__":
    # "./test_models" r"\\10.99.11.210\MeshCNN\Test_5044\stl"
    test_dir = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\three_batch_data\stl"
    save_dir = r"\\10.99.11.210\MeshCNN\MeshCNN_Train_data\three_batch_data\down_obj"

    has_pts_files = glob.glob(os.path.join(save_dir, "*.obj"))
    has_pts_file_names = [os.path.splitext(os.path.basename(pts_file))[0] for pts_file in has_pts_files]

    error_models = []

    all_stl_models = glob.glob(os.path.join(test_dir, "*.stl"))
    stl_models = []
    for stl_model in all_stl_models:
        basename = os.path.basename(stl_model)
        if basename in error_models:
            continue
        if os.path.splitext(basename)[0] in has_pts_file_names:
            continue
        stl_models.append(stl_model)
    parallel_simplify(stl_models, save_dir, 4)

    # test_model = "./test_models/7MUZ3_VS_SET_VSc2_Subsetup15_Maxillar.stl"
    # extract_gumline(test_model, save_dir)
    # show_stl_pts(test_model, os.path.splitext(os.path.basename(test_model))[0] + ".pts")


