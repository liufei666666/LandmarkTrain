# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import os,datetime
import time
from torchvision import transforms
from LMNet import LMNet14e
import numpy as np
from loaderlandmark4 import ImageList, RandomCropPos3,Randombrighten,RandomHorizontalFlip
from PIL import Image
from collections import OrderedDict


def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')

def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def imshow(inp, title=None):
    inp = inp.numpy().transpose(1, 2, 0)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.pause(10)

data_transforms = {
    'custom': transforms.Compose([
        RandomCropPos3((32,32)),
        RandomHorizontalFlip(),
        Randombrighten(),

    ]),
    'image': transforms.Compose([
        transforms.Resize((256,256)),
        #transforms.Resize((96,96)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
}

data_dir = ''

traindata_dir='/home/liufei/Documents/pupil/datalist/clpupil66.txt'
image_datasets=ImageList(fileList=traindata_dir,custom_transform = data_transforms['custom'], image_transform = data_transforms['image'], root=data_dir)
train_loader = data.DataLoader(image_datasets,batch_size=256, shuffle=True,num_workers=4, pin_memory=True)
dataset_sizes = len(image_datasets)
print(dataset_sizes)

use_gpu = torch.cuda.is_available()
#print(use_gpu)

class LossFn:
    def __init__(self, cls_factor=0.05, landmark_factorleye=1, landmark_factorreye=1,
                    landmark_factorlbrow=1, landmark_factorrbrow=1, landmark_factornose=1, landmark_factormouth=1, landmark_factorpupil=1):
    #def __init__(self, cls_factor=1):
        # loss function
        #self.cls_factor = cls_factor
        #self.box_factor = box_factor
        #self.landeye_factor = landmarkeye_factor
        #self.landbrow_factor = landmarkbrow_factor
        #self.landmark_factor = landmark_factor
        self.landmark_factorleye = landmark_factorleye
        self.landmark_factorreye = landmark_factorreye
        self.landmark_factorlbrow = landmark_factorlbrow
        self.landmark_factorrbrow = landmark_factorrbrow
        self.landmark_factornose = landmark_factornose
        self.landmark_factormouth = landmark_factormouth
        self.landmark_factorpupil = landmark_factorpupil



        #self.loss_cls = nn.CrossEntropyLoss()
        #self.loss_cls = nn.BCEWithLogitsLoss()
        #self.loss_cls = nn.MSELoss()
        #self.loss_cls = nn.CrossEntropyLoss()
        #self.  = nn.SmoothL1Loss()
        #self.loss_landmarkeye = nn.MSELoss()
        #self.loss_landmarkbrow = nn.MSELoss()
        self.loss_landmarkleye = nn.L1Loss()
        self.loss_landmarkreye = nn.L1Loss()
        self.loss_landmarklbrow = nn.L1Loss()
        self.loss_landmarkrbrow = nn.L1Loss()
        self.loss_landmarknose = nn.L1Loss()
        self.loss_landmarkmouth = nn.L1Loss()
        self.loss_landmarkpupil = nn.L1Loss()


    def cls_loss(self,gt_label,pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        #print('gt_label:',gt_label)
        mask1 = torch.eq(gt_label,0)
        mask2 = torch.eq(gt_label,1)
        mask=mask1+mask2
        #print('mask:',mask)
        chose_index = torch.nonzero(mask.data)
        #print('chose_index:',chose_index)
        if(chose_index.dim() == 0):
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)

        valid_gt_label = gt_label[chose_index]
        #print('valid_gt_label:',valid_gt_label)


        valid_pred_label = pred_label[chose_index]
        #print('valid_pred_label:',valid_pred_label)
        #print('pred_label:',pred_label)
        return self.loss_cls(valid_pred_label,valid_gt_label)*self.cls_factor

    def box_loss(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        #mask1 = torch.eq(gt_label,-1)
        mask2 = torch.eq(gt_label,1)
        mask=mask2
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        valid_pred_offset = pred_offset[chose_index,:]
        return self.loss_box(valid_pred_offset,valid_gt_offset)*self.box_factor

    def landmark_lossleye(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarkleye(valid_pred_offset,valid_gt_offset)*self.landmark_factorleye

    def landmark_lossreye(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarkreye(valid_pred_offset,valid_gt_offset)*self.landmark_factorreye

    def landmark_losslbrow(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarklbrow(valid_pred_offset,valid_gt_offset)*self.landmark_factorlbrow

    def landmark_lossrbrow(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarkrbrow(valid_pred_offset,valid_gt_offset)*self.landmark_factorrbrow

    def landmark_lossnose(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarknose(valid_pred_offset,valid_gt_offset)*self.landmark_factornose

    def landmark_lossmouth(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        mask2 = torch.eq(gt_label,0)
        mask3 = torch.eq(gt_label,-1)
        mask=mask1+mask2+mask3
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarkmouth(valid_pred_offset,valid_gt_offset)*self.landmark_factormouth

    def landmark_losspupil(self,gt_label,gt_offset,pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        mask1 = torch.eq(gt_label,1)
        
        mask=mask1
        #print(mask)
        chose_index = torch.nonzero(mask.data)
        if chose_index.dim() == 0:
            a= torch.zeros(1).cuda()
            return Variable(a)

        chose_index = torch.squeeze(chose_index)
        valid_gt_offset = gt_offset[chose_index,:]
        #print('valid_gt_offset:',valid_gt_offset)
        valid_pred_offset = pred_offset[chose_index,:]
        #print('valid_pred_offset:',valid_pred_offset)
        return self.loss_landmarkpupil(valid_pred_offset,valid_gt_offset)*self.landmark_factorpupil


def train_model(model, optimizer):
    since = time.time()
    #best_model_wts = model.state_dict()
    #best_acc = 0.0
    lossfn= LossFn()
    model.train().cuda()
    running_loss = 0.0
    running_corrects = 0
    #running_loss_cls= 0.0
    #running_loss_lm=0.0
    running_loss_lmleye=0.0
    running_loss_lmreye=0.0
    running_loss_lmlbrow=0.0
    running_loss_lmrbrow=0.0
    running_loss_lmnose=0.0
    running_loss_lmmouth=0.0
    running_loss_lmpupil=0.0
    total=0
    for idx, (inputs, labels, landmarkleye, landmarkreye, landmarklbrow, landmarkrbrow, landmarknose, landmarkmouth, landmarkpupil) in enumerate(train_loader):
    #for idx, (inputs, labels) in enumerate(train_loader):

        if use_gpu:
            inputs, labels, landmarkleye, landmarkreye, landmarklbrow, landmarkrbrow, landmarknose, landmarkmouth, landmarkpupil = Variable(inputs.cuda()), Variable(labels.cuda()), Variable(landmarkleye.cuda().float()), Variable(landmarkreye.cuda().float()), Variable(landmarklbrow.cuda().float()), Variable(landmarkrbrow.cuda().float()), Variable(landmarknose.cuda().float()), Variable(landmarkmouth.cuda().float()), Variable(landmarkpupil.cuda().float())
            #inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels, landmarkleye, landmarkreye, landmarklbrow, landmarkrbrow, landmarknose, landmarkmouth, landmarkpupil = Variable(inputs), Variable(labels), Variable(landmarkleye), Variable(landmarkreye), Variable(landmarklbrow), Variable(landmarkrbrow), Variable(landmarknose), Variable(landmarkmouth), Variable(landmarkpupil)
            #inputs, labels= Variable(inputs), Variable(labels)

        lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth,lmpupil= model(inputs)
        #lmeye,lmbrow= model(inputs)
        #print('outputs:',outputs)
        #print('torch.max:',torch.max(outputs.data, 1))
        #raw_input('pause')
        #_, preds = torch.max(outputs.data, 1)
        #preds=outputs.data>0.5

        #loss_cls = lossfn.cls_loss(labels,outputs)
        #loss_loc = lossfn.box_loss(labels,bbox,bb)
        #loss_lmeye  = lossfn.landmark_losseye(labels,landmarkeye,lmeye)
        #loss_lmbrow  = lossfn.landmark_lossbrow(labels,landmarkbrow,lmbrow)
        #loss_lm  = lossfn.landmark_loss(labels,landmark,lm)
        loss_lmleye  = lossfn.landmark_lossleye(labels,landmarkleye,lmleye)
        loss_lmreye  = lossfn.landmark_lossreye(labels,landmarkreye,lmreye)
        loss_lmlbrow  = lossfn.landmark_losslbrow(labels,landmarklbrow,lmlbrow)
        loss_lmrbrow  = lossfn.landmark_lossrbrow(labels,landmarkrbrow,lmrbrow)
        loss_lmnose  = lossfn.landmark_lossnose(labels,landmarknose,lmnose)
        loss_lmmouth  = lossfn.landmark_lossmouth(labels,landmarkmouth,lmmouth)
        loss_lmpupil  = lossfn.landmark_losspupil(labels,landmarkpupil,lmpupil)
        #print('loss_lm:',loss_lm)

        #all_loss=loss_lm
        all_loss=loss_lmleye+loss_lmreye+loss_lmlbrow+loss_lmrbrow+loss_lmnose+loss_lmmouth+loss_lmpupil
        #print(all_loss)
        #raw_input('pause')


        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()


        #running_loss_cls += loss_cls.data[0]
        #running_loss_loc += loss_loc.data[0]
        #running_loss_lmeye += loss_lmeye.data[0]
        #running_loss_lmbrow += loss_lmbrow.data[0]
        #running_loss_lm += loss_lm.data[0]
        running_loss_lmleye += loss_lmleye.data[0]
        running_loss_lmreye += loss_lmreye.data[0]
        running_loss_lmlbrow += loss_lmlbrow.data[0]
        running_loss_lmrbrow += loss_lmrbrow.data[0]
        running_loss_lmnose += loss_lmnose.data[0]
        running_loss_lmmouth += loss_lmmouth.data[0]
        running_loss_lmpupil += loss_lmpupil.data[0]
        #print(running_loss_lm)
        running_loss += all_loss.data[0]
        total+=labels.size(0)
        #print('preds:',torch.squeeze(preds))
        #print('labeldata:',labels.data)
        #raw_input('pause')
        #running_corrects += torch.sum(torch.squeeze(preds) == labels.data)
        #print('corrects:',running_corrects)
        #print('total:',total)

    #epoch_acc = running_corrects / total
    print('total:',total)
    #print('Closs: {:.4f}'.format(running_loss_cls))
    print('LEYE: {:.4f}'.format(running_loss_lmleye))
    print('REYE: {:.4f}'.format(running_loss_lmreye))
    print('LBROW: {:.4f}'.format(running_loss_lmlbrow))
    print('RBROW: {:.4f}'.format(running_loss_lmrbrow))
    print('NOSE: {:.4f}'.format(running_loss_lmnose))
    print('MOUTH: {:.4f}'.format(running_loss_lmmouth))
    print('PUPIL: {:.4f}'.format(running_loss_lmpupil))
    print('ALLloss: {:.6f}'.format(running_loss))
    #print('train_acc:{:.8f}'.format(epoch_acc))
    time_elapsed = time.time() - since
    print('Training costs {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':

    LR=1e-3

    model = LMNet14e()
    model = model.cuda()
    #model.load_state_dict(torch.load('./modelsaved/LMNet1/LMNet1b_26.pth'))
    #model.load_state_dict(torch.load('/home/ficha/Documents/riku/lm/modelsaved/LMNet5/LMNet5a_45.pth'))
    #model.load_state_dict(torch.load('/home/ficha/Documents/riku/lm/modelsaved/LMNet6/LMNet6a_200.pth'))
    #model.load_state_dict(torch.load('/home/ficha/Documents/riku/lm/modelsaved/LMNet7/LMNet7a_200.pth'))
    print('start: time={}'.format(dt()))
    #train_datalist=ImageList(fileList=traindata_dir,custom_transform = data_transforms['custom'], image_transform = data_transforms['image'], root=data_dir)
    #train_loader = data.DataLoader(train_datalist,batch_size= 256, shuffle=True ,num_workers=4, pin_memory=True)
    #optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=LR)
    #optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)

    for epoch in range(1,500):
        if epoch == 25: LR *=0.1
        if epoch == 50: LR *=0.1
        if epoch == 75: LR *=0.1
        if epoch == 100: LR *=0.1

        print('Epoch={}'.format(epoch))
        print('-' * 10)
        print('start: time={}'.format(dt()))
        train_model(model, optimizer)
        if epoch %5==0:
            save_model(model,'./modelsaved/LMNet14e/LMNet14'+'_{}.pth'.format(epoch))
        print('-' * 10)
