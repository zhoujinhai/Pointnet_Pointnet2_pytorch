import aiohttp
from aiohttp import formdata
import asyncio
import os
import time
import json
import numpy as np


def show_pcl_data(data, label_cls=-1):
    import vedo
    points = data[:, 0:3]

    colours = ["grey", "red", "blue", "brown", "yellow", "green", "black", "pink"]
    labels = data[:, label_cls]  # 最后一列为标签列
    diff_label = np.unique(labels)
    group_points = []
    for label in diff_label:
        point_group = points[labels == label]
        group_points.append(point_group)

    show_pts = []
    for i, point in enumerate(group_points):
        pt = vedo.Points(point.reshape(-1, 3)).pointSize(6).c((colours[i % len(colours)]))  # 显示点
        show_pts.append(pt)
    vedo.show(show_pts)


async def do_recognize(web_ip, web_port, file_path):
    try:
        codename = os.path.basename(file_path)
        print('filename: {}'.format(file_path))
        # file_data = {"file": open(file_path, "rb")}
        file_data = formdata.FormData()
        file_data.add_field('file',
                       open(file_path, 'rb'),
                       # content_type="multipart/form-data; boundary=--dd7db5e4c3bd4d5187bb978aef4d85b1",
                       filename=codename)
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    url='http://{}:{}/recognize'.format(web_ip, web_port),
                    data=file_data,
                    # headers={'Content-Type': 'multipart/form-data; boundary=--dd7db5e4c3bd4d5187bb978aef4d85b1'}
            ) as resp:
                respond = await resp.text()
                respond = respond.replace('\\"', '"')
                respond = respond[1:-1]   # remove " at the begin and end
                result = json.loads(respond)
                idx = result.get("text")
                status = result.get("returnCode")
                print("predict {}, res: {}".format(status, idx))

                # show result
                data = np.loadtxt(file_path, skiprows=10).astype(np.float32)
                label = [0] * len(data)
                for r in idx:
                    label[r] = 1
                show_data = np.c_[data, np.asarray(label)]
                show_pcl_data(show_data)
    except:
        print("do_recognize Error {} \n".format(file_path))


def run_test(web_ip, web_port, filename, loop):
    try:
        loop.run_until_complete(do_recognize(web_ip, web_port, filename))
    except:
       print ("run_test Error")


if __name__ == '__main__':
    file_dir = r"./test_datas"
    filenames = os.listdir(file_dir)
    files = [os.path.join(file_dir, filename) for filename in filenames]
    ip = "127.0.0.1"   # "192.168.102.116"
    port = 8000
    start = time.time()
    try:
        event_loop = asyncio.get_event_loop()
        tasks = [run_test(ip, port, filename, event_loop) for filename in files]
        event_loop.close()
    except:
        print("__main__ Error")
    end = time.time()
    print("run time is : {}s".format(end - start))

