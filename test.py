import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import os
import cv2
from LMNet import LMNet14e


trained_netfile = "/home/liufei/Documents/pupil/modelsaved/LMNet14e/LMNet14_480.pth"
image_file = "/home/liufei/Documents/pupil/infrared_15_1684.bmp"
net = LMNet14e()
net.load_state_dict(torch.load(trained_netfile))
net.eval()
image = cv2.imread(image_file)
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
image = cv2.resize(image,(256,256))
#cv2.imshow('copy_im',image)
#cv2.waitKey()
image = (image-127.5)*0.0078125 # [0,255] -> [-1,1]
image = np.expand_dims(image,axis=0)
image = np.array([image], dtype = np.float)
inputs = torch.from_numpy(image).float()
#print(inputs)
inputs = Variable(inputs)
lmleye,lmreye,lmlbrow,lmrbrow,lmnose,lmmouth,lmpupil= net(inputs)
print(lmpupil[0][1])
type(lmpupil)
