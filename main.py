from config import opt
import os
import torch as t
import models
from data.data_load import Data_Load
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm
from torchvision import transforms as T

from utils.loss import calc_loss


def train(**kwargs):
    opt._parse(kwargs)
    # vis = visualizer(opt.env, port=opt.vis_port)

    #######################################配置模型

    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    ######################################### 导入数据
    transform = T.Compose([
        T.Resize(300),  # 将最短resize至400，长宽比不变
        T.CenterCrop(256),  # 将中间大小400*400裁剪
        T.ToTensor(),
        T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 标准化至[-1,1]
    ])
    train_data = Data_Load(opt.image_root, opt.label_root,transforms=None)
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)


    #################################### 损失函数与优化
    #criterion = calc_loss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # 计算误差
    loss_meter = meter.AverageValueMeter()
    # confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10

    ##################################train
    for epoch in range(opt.max_epoch):
        loss_epoch = 0

        loss_meter.reset()

        for i, (data, label) in tqdm(enumerate(train_dataloader)):
            # train model
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            prediction = model(input)
            loss = calc_loss(prediction, target)
            loss.backward()
            optimizer.step()

            # meters update and visualize
            loss_meter.add(loss.item())
            # detach 一下更安全保险
            loss_epoch = loss_epoch + loss.detach().numpy()
        print("epoch=", epoch, "Loss=", loss_epoch / 600)

        model.save()

        # update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = loss_meter.value()[0]



if __name__ == '__main__':
    train()