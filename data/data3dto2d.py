import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import SimpleITK as sitk

class Data3d_to_2d():
    def __init__(self, dataset_path, datasave_path):
        # 放置所有.nii.gz的文件路径
        self.datasave_path = datasave_path  # 存储路径
        img_niigzs_name = os.listdir(dataset_path)
        self.num = len(img_niigzs_name)
        self.img_niigzs_path = []

        # 将所有的nii.gz文件的路径存储在一个list中
        for name in img_niigzs_name:
            self.img_niigzs_path.append(dataset_path + "/" + name)

    # 读取单个nii.gz文件的data,shape=[12,880,880]
    def dataread(self, niigz_path):
        data_raw = sitk.ReadImage(niigz_path)
        data = sitk.GetArrayFromImage(data_raw)
        return data

    # 保存一个nii.gz的转换图片
    def datasave(self, data, file):
        for i in range(data.shape[0]):
            plt.imsave(file + "/" + str(i + 1) + ".jpg", data[i])

    def img_save(self):
        for i in range(self.num):
            img_niigzs_path = self.img_niigzs_path[i]
            data = self.dataread(self.img_niigzs_path[i])
            file = self.datasave_path + "/" + "Case" + str(i + 1)
            os.makedirs(file)
            self.datasave(data, file)


file="D:/segmentation/dataset/train/label2d"
dataset_path="D:/segmentation/dataset/train/label"
data=Data3d_to_2d(dataset_path,file).img_save()