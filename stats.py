import os
import time
# s = os.listdir('/home/student2021/MMD/dataset/images')
#
# dict = {}
# for i in range(1000):
#     dict[s[i]] = i
#
#
#
# # 记时开始
# start_time = time.time()
#
# for i in range(1000):
#     z = dict[s[i]]
#
# # 记时结束
# end_time = time.time()
#
# # 计算运行时长
# elapsed_time = end_time - start_time
# print(f"程序运行时长：{elapsed_time}秒")
# print(len(s))
import torch
x = torch.load(DatasetConfig.stage1_model_dir)['state_dict']