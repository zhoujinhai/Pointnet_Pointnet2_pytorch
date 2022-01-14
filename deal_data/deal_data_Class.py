#!/usr/bin/env python
# coding: utf-8
import numpy as np
import os
import queue
from functools import reduce
import glob
from shutil import copyfile, move
from multiprocessing import Process
from tqdm import tqdm


class GenPcdFile(object):
    """
    根据obj和pts文件生成pcd文件，用于训练  格式为 x y z pt_label n ==> xyz表示坐标， pt_label表示该点的标签 n表示该标签的第几个实例 默认为-1
    """
    def __init__(self, obj_path, gum_line_path, save_path="./pcd/"):
        self.obj_path = obj_path
        self.gum_line_path = gum_line_path
        self.basename = os.path.splitext(os.path.basename(self.obj_path))[0]
        self.save_path = save_path
        self.n_classes = 2
        self.step = 6                        # 原始牙龈点挑选步长
        self.max_cycley_len = 6              # 闭环容许最小长度
        self.vs = None
        self.faces = None
        self.edges = None
        self.gemm_edges = None               # 邻边id
        self.gum_line_pts = None             # 牙龈线点
        self.gum_line_ids = None             # 牙龈线点对应的id
        self.face_labels = None              # 面的标记
        self.pts_labels = None               # 0:牙龈线点  1:牙龈  2:牙齿
        
        # 如果文件不存在则检查路径，路径不存在先创建
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        
    def generate_file(self):
        # 获取模型信息
        self.get_obj_info()
        
        # 获取牙龈线信息
        self.get_gum_line_pts()
        
        # 牙龈线点投影
        self.gum_line_projection(self.step)
        
        # 连通牙龈线点
        self.connect_line_points()
        
        # 去掉闭环
        self.drop_cycle(self.max_cycley_len)
        
        # 标记面
        self.label_faces()

        # 根据个数判断
        self.judge_by_zaxis()

        # 标记点
        self.label_vs()

        # 保存文件
        # np.set_printoptions(threshold=np.inf)
        # print("face labels: \n", len(self.face_labels), self.face_labels, self.pts_labels)
        self.save_pcd()
    
    # ----------- 1、解析文件 ----------------
    def get_obj_info(self):
        """
        解析obj文件，得到对应的vs和faces
        """
        vs, faces = [], []
        f = open(self.obj_path)
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif splitted_line[0] == 'f':
                face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
                assert len(face_vertex_ids) == 3, "一个面最多三个顶点"
                # 下标从0开始
                face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                                   for ind in face_vertex_ids]
                faces.append(face_vertex_ids)
        f.close()
        vs = np.asarray(vs)
        faces = np.asarray(faces, dtype=int)
        assert np.logical_and(faces >= 0, faces < len(vs)).all()
        
        self.vs = vs
        self.faces = faces
        
    def get_gum_line_pts(self):
        """
        读取牙龈线文件，pts格式
        """
        """
           读取牙龈线文件，pts格式
           """
        f = open(self.gum_line_path)
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

        # f = open(self.gum_line_path)
        # # print(self.gum_line_path)
        # pts = []
        # for line in f:
        #     line = line.strip()
        #     splitted_line = line.split()
        #     point = [float(i) for i in splitted_line]
        #     assert len(point) == 3, "点的坐标为x,y,z"
        #     pts.append(point)
        #
        # f.close()
        # pts = np.asarray(pts)
        self.gum_line_pts = pts
    
    # ----------- 2、牙龈线投影 ----------------
    @staticmethod
    def find_nearest_vector(array, value):
        """
        从列表array中查找和value最近的值及对应的下标
        @array: 列表         N*M: N行M列
        @value: 需要查找的值  K*1*M: K个1行M列的值
        """
        idx = np.array([np.linalg.norm(x, axis=1) for x in array-value])
        idx = np.argmin(idx, axis=1)
        return array[idx], idx
    
    def gum_line_projection(self, step=6):
        """
        找到原始模型牙龈线投影到新模型后的点
        @step: 对原始点每step个取一次
        """
        mesh_pts = []
        ids = []
        for pt in self.gum_line_pts[::step]:
            nearest_pt, idx = self.find_nearest_vector(self.vs, pt.reshape(1, 1, 3))
            mesh_pts.append(nearest_pt)
            ids.append(idx)
            
        mesh_pts.append(mesh_pts[0])  # 保证闭合
        ids.append(ids[0])
        mesh_pts = np.asarray(mesh_pts).reshape(-1, 3)
        ids = np.asarray(ids).reshape(-1, 1)
        
        self.gum_line_pts = mesh_pts
        self.gum_line_ids = ids
    
    # ----------- 3、牙龈线后处理 ----------------
    def get_dist_p2p(self, pt1, pt2):
        """
        根据两点id，确定其距离，为了减少计算，这里未开根号
        @vs: 点集 [[x, y, z], ...]
        @pt1: 第一个点的id
        @pt2: 第二个点的id
        return: 两点之间距离
        """
        point1 = self.vs[pt1]
        point2 = self.vs[pt2]
        diff = point1 - point2
        dist = reduce(lambda x, y: x+y, (map(lambda x: x ** 2, diff)))
        return dist

    def get_dist_path(self, path):
        """
        找出一条路径的距离
        @vs: 点集 [[x, y, z], ...]
        @path: 路径 [pt1, pt2, ...]
        return: 路径长度
        """
        sum_dist = 0
        for i in range(len(path)-1):
            prev = path[i]
            last = path[i+1]
            dist = self.get_dist_p2p(prev, last)
            sum_dist += dist
        return sum_dist
    
    def find_nearest_points(self, point_id):
        """
        根据点的id值和所有面查找出和该点相邻的所有点
        @self.faces: 所有三角面，N*3, 每个值为点的id
        @point_id: 点的id值
        """
        nearest_faces = self.faces[np.argwhere(self.faces==point_id)[:, 0]]   # 第一个为行数，第二个为该点出现的位置
        nearest_points = list(set(nearest_faces.reshape(1, -1)[0]))           # 一个二维数组[[...]]
        nearest_points.remove(point_id)                                       # 删除自身
        return nearest_points
    
    @staticmethod
    def find_path_by_bfs(checked_dict, start, end):
        """
        根据起点和终点，以及检查过的点，利用BFS找出一条连通路径
        @checked_dict: 查找过的点 {“A": ["A1", "A2", ...], ...}
        @start: 起始点对应的id值
        @end: 终点对应的id值
        return: 按起点到终点的顺序返回所有点 [start, "A", "B", ..., end]
        """
        find_res = [end]
        keys = checked_dict.keys()
        # print(keys)
        count = 0
        while end != start and count < len(checked_dict):
            # print(count, end)
            for key in keys:
                if end in checked_dict[key]:
                    find_res.append(key)
                    end = key
                    break

            count += 1
        if count > len(checked_dict):
            print("not found!")

        find_res.reverse()
        return find_res
    
    # 3.1 保证牙龈线连通性
    def connect_line_points(self):
        """
        保证投影后的点连通
        @pt_ids: 点的id
        @faces: 模型的面
        """
        new_ids = []
        i = 0 
        new_ids.append(self.gum_line_ids[0])
        while i < (len(self.gum_line_ids)-1):
            start = self.gum_line_ids[i][0]
            end = self.gum_line_ids[i+1][0]

            start_nearest_points = self.find_nearest_points(start)

            if end == start:
                i += 1
            elif end in start_nearest_points:
                i += 1
                new_ids.append([end])
            else:
                checked = {}                   # 已经查找的
                checked[start] = start_nearest_points
                all_path = []
                waiting_queue = queue.Queue()  # 等待队列
                for item in start_nearest_points:
                    waiting_queue.put(item)

                while not waiting_queue.empty():
                    check_id = waiting_queue.get()
                    if check_id not in checked:
                        check_nearest_points = self.find_nearest_points(check_id)
                        # 找到
                        if end in check_nearest_points:  
                            path_list = self.find_path_by_bfs(checked, start, check_id)
                            path_list.append(end)
                            path_length = len(path_list)
                            all_path.append(path_list)
                            # print(path_list, check_id, start, end)
                            
                            # 任意找一条路径
                            # for item in path_list[1:]:
                            #     new_ids.append([item])
                            # break
                            
                            # 存在则找出剩下路径, 后面从里挑选最短路径
                            checked[check_id] = check_nearest_points
                            while not waiting_queue.empty():
                                check_id = waiting_queue.get()
                                if check_id not in checked:
                                    check_nearest_points = self.find_nearest_points(check_id)
                                    if end in check_nearest_points:  
                                        path_list = self.find_path_by_bfs(checked, start, check_id)
                                        path_list.append(end)
                                        # 顺序原因，前面节点数小于后面节点数 即领域问题
                                        if len(path_list) > path_length:
                                            continue
                                        all_path.append(path_list)
                                        checked[check_id] = check_nearest_points

                            # 通过距离找到最短路径
                            min_dist = float('inf')
                            min_idx = -1
                            for idx, path in enumerate(all_path):
                                dist = self.get_dist_path(path)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = idx
                            short_path = all_path[min_idx]
                            for item in short_path[1:]:
                                new_ids.append([item])
                            
                        else:
                            for item in check_nearest_points:
                                waiting_queue.put(item)
                            checked[check_id] = check_nearest_points
                i += 1
                
        self.gum_line_ids = new_ids
    
    # 3.2 去掉牙龈线闭环
    def drop_cycle(self, max_length=6):
        """
        删除列表中形成的小闭环
        @max_length: 容许闭环的最小长度
        return: 输出删除小闭环后的列表
        """
        drop_list = []
        if isinstance(self.gum_line_ids, np.ndarray):
            self.gum_line_ids = self.gum_line_ids.tolist()
        for i, item in enumerate(self.gum_line_ids):
            if item not in drop_list:
                drop_list.append(item)
            else:
                first_index = self.gum_line_ids.index(item)   # item第一次出现的位置
                if i - first_index < max_length:
                    idx = drop_list.index(item)    # item在drop_list中的位置，用于剔除后面添加的
                    drop_list = drop_list[:idx+1]
                else:
                    drop_list.append(item)
        drop_list = np.asarray(drop_list)

        self.gum_line_ids = drop_list
        self.gum_line_pts = self.vs[drop_list]
        
    # ----------- 4、根据牙龈线对面进行标记 ----------------
    def find_faces_by_2point(self, id1, id2):
        """
        根据两个点确定以两点所在边为公共边的两个面
        @self.faces: 所有面，N*3, 值表示点的id值
        @id1: 第一个点的id值
        @id2: 第二个点的id值
        return: 2*3, [面的id，第一个点的位置， 第二个点的位置]
        """
        p1_faces = np.argwhere(self.faces == id1)  # 行id, 列id
        p2_faces = np.argwhere(self.faces == id2)

        intersection_faces = []
        for val1 in p1_faces:
            for val2 in p2_faces:
                if val1[0] == val2[0]:
                    intersection_faces.append([val1[0], val1[1], val2[1]])

        return intersection_faces
    
    def find_nearest_faces(self, face_id):
        """
        根据面的id值，找出相邻三个面的id值
        @self.faces: 所有面，行数代表其id
        @face_id: 需要查找邻面的id
        return: 返回其他三个面的id值
        """
        face_ids = []
        face = self.faces[face_id]
        face = np.append(face, face[0])
        for i in range(len(face)-1):
            face_i = self.find_faces_by_2point(face[i], face[i+1])

            if len(face_i) > 0 and face_i[0][0] != face_id:
                face_ids.append(face_i[0][0])
            else:
                if len(face_i) > 1:
                    face_ids.append(face_i[1][0])
        return face_ids
    
    def label_faces(self):
        """
        根据牙龈线点 利用区域增广方式对面进行标注
        @self.faces: 需要标注的面
        @self.gum_line_ids: 两两连通的牙龈线点
        return: 面的标签
        """
        # ### ---- init ----
        face_labels = [-1] * len(self.faces)
        is_labeled = [False] * len(self.faces)
        label_1 = [[0, 1], [1, 2], [2, 0]]  # label_2 = [[0, 2], [2, 1], [1, 0]]
        seed_faces_q = queue.Queue()
        # print(len(self.faces[is_labeled]))

        # ### ---- 遍历牙龈点 根据顺时针或逆时针确定种子面 ----
        for i in range(len(self.gum_line_ids) - 1):
            pt1 = self.gum_line_ids[i]
            pt2 = self.gum_line_ids[i+1]
            find_faces = self.find_faces_by_2point(pt1, pt2)
            # print("---------")
            for face in find_faces:
                face_id = face[0]
                # print(face)
                if face[1:] in label_1:
                    face_labels[face_id] = 1
                else:
                    face_labels[face_id] = 2

                is_labeled[face_id] = True
                seed_faces_q.put(face_id)

        # 查看牙龈线分割情况
        # save_model_part(down_obj_vs, down_obj_faces, face_labels)
        # print(len(self.faces[is_labeled]))

        # ### ---- 根据种子面继续增广 ----
        remind_faces = len(self.faces) - len(self.faces[is_labeled])
        while (not seed_faces_q.empty()) and remind_faces > 0:
            face_id = seed_faces_q.get()
            face_label = face_labels[face_id]
            nearest_face_ids = self.find_nearest_faces(face_id)
            for idx in nearest_face_ids:
                if is_labeled[idx] is False:
                    face_labels[idx] = face_label
                    is_labeled[idx] = True
                    seed_faces_q.put(idx)

            remind_faces = len(self.faces) - len(self.faces[is_labeled])
        face_labels = np.asarray(face_labels)

        self.face_labels = face_labels

    # ----------- 5、根据区域点z轴坐标平均值挑选牙龈 ----------------
    def judge_by_zaxis(self):
        """
        牙龈所在区域z最小，除去该部分其他皆为牙齿
        """
        last_labels = np.array(self.face_labels)
        n_classes = self.face_labels.max()
        avg_z = [0] * n_classes

        for i in range(n_classes):
            faces_class = self.faces[self.face_labels == i + 1]
            for face in faces_class:
                face_z_sum = self.vs[face[0]][2] + self.vs[face[1]][2] + self.vs[face[2]][2]
                avg_z[i] += face_z_sum
            avg_z[i] /= len(faces_class)
        # print(avg_z)
        # min z
        gingiva_label = avg_z.index(min(avg_z)) + 1
        # print(gingiva_label)

        # change label
        last_labels[self.face_labels == gingiva_label] = 1  # 牙龈记为1
        last_labels[self.face_labels != gingiva_label] = 2  # 牙齿记为2
        self.face_labels = last_labels

    # ----------- 6、根据面标签对点进行标记 ----------------
    def label_vs(self):
        pt_labels = [0] * len(self.vs)
        for idx, face in enumerate(self.faces):
            label = self.face_labels[idx]
            for pt_id in face:
                pt_labels[pt_id] = label
        for idx in self.gum_line_ids:
            pt_labels[idx[0]] = 0
        self.pts_labels = np.asarray(pt_labels)

    # ----------- 7、保存pcd文件 ------------
    def save_pcd(self):
        n = len(self.vs)
        pcd_path = os.path.join(self.save_path, self.basename + ".pcd")
        with open(pcd_path, mode='w') as f:
            # write head
            f.write("VERSION 0.7\nFIELDS x y z _\nSIZE 4 4 4 1\nTYPE F F F U\nCOUNT 1 1 1 4\n")
            f.write("WIDTH {}\n".format(n))
            f.write("HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
            f.write("POINTS {}\n".format(n))
            f.write("DATA ascii\n")

            for idx, pt in enumerate(self.vs):
                f.write("{} {} {} {} {}\n".format(pt[0], pt[1], pt[2], self.pts_labels[idx], self.pts_labels[idx] - 1))

    # ---------- 7、show pcd file ----------
    def show_pcl_data(self):
        import vedo
        points = self.vs
        labels = self.pts_labels
        colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]

        diff_label = np.unique(labels)
        print("res_label: ", diff_label)
        group_points = []
        for label in diff_label:
            point_group = points[labels == label]
            group_points.append(point_group)

        show_pts = []
        for i, point in enumerate(group_points):
            pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
            show_pts.append(pt)
        vedo.show(show_pts)


def generate_batch(obj_list, pts_list, save_path, pts_path, obj_path):
    for i in tqdm(range(len(obj_list))):
        obj_file = obj_list[i]
        obj_base_name = os.path.basename(obj_file)

        pts_base_name = obj_base_name.replace(".obj", ".pts")
        pts_file_path = os.path.join(pts_path, pts_base_name)
        obj_file_path = os.path.join(obj_path, obj_base_name)

        if pts_file_path in pts_list:
            genPcdFile = GenPcdFile(obj_file_path, pts_file_path, save_path)
            genPcdFile.generate_file()


def parallel_generate(obj_list, pts_list, save_path, pts_path, obj_path, n_workers=8):
    """
    多进程处理
    """
    if len(obj_list) < n_workers:
        n_workers = len(obj_list)
    chunk_len = len(obj_list) // n_workers

    chunk_lists = [obj_list[i:i+chunk_len] for i in range(0, (n_workers-1)*chunk_len, chunk_len)]
    chunk_lists.append(obj_list[(n_workers - 1)*chunk_len:])
    
    process_list = [Process(target=generate_batch, args=(chunk_list, pts_list, save_path, pts_path, obj_path)) for chunk_list in chunk_lists]
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()


# In[16]:

if __name__ == "__main__":
    # # ------------ Test one ----------------
    # # obj path
    # obj_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\down_obj\1.obj"
    # # gingiva path
    # ginggiva_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pts\1.pts"
    #
    # genPcdFile = GenPcdFile(obj_path, ginggiva_path)
    # genPcdFile.generate_file()
    # genPcdFile.show_pcl_data()

    # # ------------ 批量生成  ----------------
    pts_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pts"
    obj_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\down_obj"
    save_path = r"\\10.99.11.210\MeshCNN\pointCloudData\GumLine\pcd"

    pts_lists = glob.glob(os.path.join(pts_path, "*.pts"))
    obj_lists = glob.glob(os.path.join(obj_path, "*.obj"))
    print("all pts: ", len(pts_lists), " obj: ", len(obj_lists))

    # 跳过处理过的数据
    obj_list = []
    pts_list = []
    for obj_file in obj_lists:
        basename = os.path.splitext(os.path.basename(obj_file))[0]
        pcd_file = os.path.join(save_path, basename + ".pcd")
        if os.path.isfile(pcd_file):
            continue
        obj_list.append(obj_file)
        pts_list.append(os.path.join(pts_path, basename + ".pts"))
    print("need deal pts: ", len(pts_list), " obj: ", len(obj_list))

    # 如果文件不存在则检查路径，路径不存在先创建路径
    parallel_generate(obj_list, pts_list, save_path, pts_path, obj_path, 4)






