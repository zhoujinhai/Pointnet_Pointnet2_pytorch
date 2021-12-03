import cv2
import numpy as np
import torch
import onnx
import time


# # register custom ops
# class CustomLayer(object):
#     def __init__(self, params, blobs):
#         self.channel = 0
#         self.nPoints = 0
#
#     def getMemoryShapes(self, inputs):
#         print("inputs: ", inputs)
#         points, cate = inputs[0], inputs[1]
#         n, c = points[0], points[1]
#         print("n", n, "c", c)
#
#         self.channel = c
#         self.nPoints = n
#
#     def forward(self, inputs):
#         print("inputs: ", inputs)
#         print("inputs[0] shape: ", inputs[0].shape)
#         return inputs[0]
#
#
# cv2.dnn_registerLayer("test_custom", CustomLayer)


# class ConstantOfShapeLayer(object):
#     def __init__(self, params, blobs):
#         self.value = blobs[0]
#         self.shape = blobs[1]
#
#     def getMemoryShapes(self, inputs):
#         return [self.shape]
#
#     def forward(self, inputs):
#         return [np.ones(self.shape) * self.value]
#
#
# cv2.dnn_registerLayer("ConstantOfShape", ConstantOfShapeLayer)   # https://github.com/opencv/opencv/issues/16662


if __name__ == "__main__":
    onnx_path = "./model_ori.onnx"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    with open("./model_graph.txt", "w") as f:
        print("save model_graph...")
        f.write(onnx.helper.printable_graph(onnx_model.graph))
    start = time.time()
    net = cv2.dnn.readNetFromONNX(onnx_path)
    end = time.time()
    print("read model Done, time: ", end - start)
    points = torch.randn((1, 3, 6000)).numpy().astype(np.float32)
    cate = torch.ones((1, 1, 1)).numpy().astype(np.float32)

    print("start set input")
    net.setInput(points, name="points")
    print("set input 1 Done")
    net.setInput(cate, name="cate")
    print("set input 2 Done")
    print("res: ", net.forward())