#coding:utf8

import torch

class Default_Config(object):
    env="default"
    vis_port=8097
    model="U_net"

    image_root="D:\segmentation\dataset\\train\image2d"
    label_root="D:\segmentation\dataset\\train\label2d"
    #test_data_root="./data/test"
    #load_model_path="D:\\SIGS\\ultradata_classifier\\checkpoints\\Alexnet_11_33.pth"
    load_model_path =None
    chass_rawdata_root="D:\\SIGS\\ultradata_Dong\\Part1"
    head_path = "D:\\SIGS\\ultradata_classifier\\head\\"

    batch_size=2
    use_gpu=False
    num_workers=0

    max_epoch=100
    lr=0.001
    lr_decay=0.5
    weight_decay = 0e-5

    print_freq=20

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        #opt.device = torch.device('cuda') if opt.use_gpu else torch.device('cpu')
        opt.device = torch.device('cpu')
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt=Default_Config()