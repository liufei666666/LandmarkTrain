from PIL import Image
import numpy as np
import torch.utils.data as data
import os
import os.path
import torch
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import random
import cv2
import numpy.random as npr
from PIL import ImageDraw
from math import sqrt

def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,(box_area + area - inter))
    #ovr = inter / (box_area + area - inter)
    return ovr

def IoU2(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = np.true_divide(inter,area)
    #ovr = inter / (box_area + area - inter)
    return ovr



class Randombrighten(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.3):
        self.p = p
        self.delta=60/255

    def __call__(self, sample):
        #image, target, path =sample['image'],sample['target'],sample['path']
        image, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth,landmarkpupil, path = sample['image'],sample['target'],sample['landmarkleye'], sample['landmarkreye'],sample['landmarklbrow'], sample['landmarkrbrow'],sample['landmarknose'], sample['landmarkmouth'], sample['landmarkpupil'],sample['path']
        #print(target)
        """
        Args:
        img (PIL Image): Image with anotations to be flipped.
        Returns:
            PIL Image: Randomly flipped image with new anotations.
        """


        if random.random() < 0.3:
            #image.show()
            a= npr.uniform(0.7,1.5)
            #image = transforms.ColorJitter(brightness=self.delta)(image)
            image = transforms.functional.adjust_brightness(image,a)
            #image.show()
        #a= npr.uniform(0.7,1.5)
        if random.random() > 0.7:
            a= npr.uniform(0.7,1.5)
            image = transforms.functional.adjust_contrast(image,a)

        '''
        if random.random() < 0.5:
            image.show()
            #image = transforms.ColorJitter(brightness=self.delta)(image)
            image = transforms.functional.adjust_contrast(image,1.5)
            image.show()
        '''


        return {'image': image, 'target':target, 'landmarkleye':landmarkleye,'landmarkreye':landmarkreye,'landmarklbrow':landmarklbrow,'landmarkrbrow':landmarkrbrow,'landmarknose':landmarknose,'landmarkmouth':landmarkmouth,'landmarkpupil':landmarkpupil,'path': path}


    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        #image, target, landmark, path = sample['image'],sample['target'],sample['landmark'],sample['path']
        image, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth,landmarkpupil, path = sample['image'],sample['target'],sample['landmarkleye'], sample['landmarkreye'],sample['landmarklbrow'], sample['landmarkrbrow'],sample['landmarknose'], sample['landmarkmouth'],sample['landmarkpupil'],sample['path']

        """
        Args:
        img (PIL Image): Image with anotations to be flipped.
        Returns:
            PIL Image: Randomly flipped image with new anotations.
        """

        if (target== 0 or target== 1 or target==-1) and random.random() < self.p:
        #if  target==-1 and random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            landmarknewleye=np.ones([20])
            landmarknewreye=np.ones([20])
            landmarknewlbrow=np.ones([16])
            landmarknewrbrow=np.ones([16])
            landmarknewnose=np.ones([20])
            landmarknewmouth=np.ones([36])
            landmarknewpupil=np.ones([4])

            landmarknewleye[0],landmarknewleye[0+10]=1-landmarkreye[4],landmarkreye[10+4]
            landmarknewleye[1],landmarknewleye[1+10]=1-landmarkreye[3],landmarkreye[10+3]
            landmarknewleye[2],landmarknewleye[2+10]=1-landmarkreye[2],landmarkreye[10+2]
            landmarknewleye[3],landmarknewleye[3+10]=1-landmarkreye[1],landmarkreye[10+1]
            landmarknewleye[4],landmarknewleye[4+10]=1-landmarkreye[0],landmarkreye[10+0]
            landmarknewleye[5],landmarknewleye[5+10]=1-landmarkreye[7],landmarkreye[10+7]
            landmarknewleye[6],landmarknewleye[6+10]=1-landmarkreye[6],landmarkreye[10+6]
            landmarknewleye[7],landmarknewleye[7+10]=1-landmarkreye[5],landmarkreye[10+5]
            landmarknewleye[8],landmarknewleye[8+10]=1-landmarkreye[8],landmarkreye[10+8]
            landmarknewleye[9],landmarknewleye[9+10]=1-landmarkreye[9],landmarkreye[10+9]

            landmarknewreye[0],landmarknewreye[0+10]=1-landmarkleye[4],landmarkleye[10+4]
            landmarknewreye[1],landmarknewreye[1+10]=1-landmarkleye[3],landmarkleye[10+3]
            landmarknewreye[2],landmarknewreye[2+10]=1-landmarkleye[2],landmarkleye[10+2]
            landmarknewreye[3],landmarknewreye[3+10]=1-landmarkleye[1],landmarkleye[10+1]
            landmarknewreye[4],landmarknewreye[4+10]=1-landmarkleye[0],landmarkleye[10+0]
            landmarknewreye[5],landmarknewreye[5+10]=1-landmarkleye[7],landmarkleye[10+7]
            landmarknewreye[6],landmarknewreye[6+10]=1-landmarkleye[6],landmarkleye[10+6]
            landmarknewreye[7],landmarknewreye[7+10]=1-landmarkleye[5],landmarkleye[10+5]
            landmarknewreye[8],landmarknewreye[8+10]=1-landmarkleye[8],landmarkleye[10+8]
            landmarknewreye[9],landmarknewreye[9+10]=1-landmarkleye[9],landmarkleye[10+9]

            landmarknewlbrow[0],landmarknewlbrow[0+8]=1-landmarkrbrow[4],landmarkrbrow[8+4]
            landmarknewlbrow[1],landmarknewlbrow[1+8]=1-landmarkrbrow[3],landmarkrbrow[8+3]
            landmarknewlbrow[2],landmarknewlbrow[2+8]=1-landmarkrbrow[2],landmarkrbrow[8+2]
            landmarknewlbrow[3],landmarknewlbrow[3+8]=1-landmarkrbrow[1],landmarkrbrow[8+1]
            landmarknewlbrow[4],landmarknewlbrow[4+8]=1-landmarkrbrow[0],landmarkrbrow[8+0]
            landmarknewlbrow[5],landmarknewlbrow[5+8]=1-landmarkrbrow[7],landmarkrbrow[8+7]
            landmarknewlbrow[6],landmarknewlbrow[6+8]=1-landmarkrbrow[6],landmarkrbrow[8+6]
            landmarknewlbrow[7],landmarknewlbrow[7+8]=1-landmarkrbrow[5],landmarkrbrow[8+5]

            landmarknewrbrow[0],landmarknewrbrow[0+8]=1-landmarklbrow[4],landmarklbrow[8+4]
            landmarknewrbrow[1],landmarknewrbrow[1+8]=1-landmarklbrow[3],landmarklbrow[8+3]
            landmarknewrbrow[2],landmarknewrbrow[2+8]=1-landmarklbrow[2],landmarklbrow[8+2]
            landmarknewrbrow[3],landmarknewrbrow[3+8]=1-landmarklbrow[1],landmarklbrow[8+1]
            landmarknewrbrow[4],landmarknewrbrow[4+8]=1-landmarklbrow[0],landmarklbrow[8+0]
            landmarknewrbrow[5],landmarknewrbrow[5+8]=1-landmarklbrow[7],landmarklbrow[8+7]
            landmarknewrbrow[6],landmarknewrbrow[6+8]=1-landmarklbrow[6],landmarklbrow[8+6]
            landmarknewrbrow[7],landmarknewrbrow[7+8]=1-landmarklbrow[5],landmarklbrow[8+5]

            landmarknewnose[0],landmarknewnose[0+10]=1-landmarknose[8],landmarknose[10+8]
            landmarknewnose[1],landmarknewnose[1+10]=1-landmarknose[7],landmarknose[10+7]
            landmarknewnose[2],landmarknewnose[2+10]=1-landmarknose[6],landmarknose[10+6]
            landmarknewnose[3],landmarknewnose[3+10]=1-landmarknose[5],landmarknose[10+5]
            landmarknewnose[4],landmarknewnose[4+10]=1-landmarknose[4],landmarknose[10+4]
            landmarknewnose[5],landmarknewnose[5+10]=1-landmarknose[3],landmarknose[10+3]
            landmarknewnose[6],landmarknewnose[6+10]=1-landmarknose[2],landmarknose[10+2]
            landmarknewnose[7],landmarknewnose[7+10]=1-landmarknose[1],landmarknose[10+1]
            landmarknewnose[8],landmarknewnose[8+10]=1-landmarknose[0],landmarknose[10+0]
            landmarknewnose[9],landmarknewnose[9+10]=1-landmarknose[9],landmarknose[10+9]


            landmarknewmouth[0],  landmarknewmouth[0+18]=1-landmarkmouth[6],landmarkmouth[18+6]
            landmarknewmouth[1],  landmarknewmouth[1+18]=1-landmarkmouth[5],landmarkmouth[18+5]
            landmarknewmouth[2],  landmarknewmouth[2+18]=1-landmarkmouth[4],landmarkmouth[18+4]
            landmarknewmouth[3],  landmarknewmouth[3+18]=1-landmarkmouth[3],landmarkmouth[18+3]
            landmarknewmouth[4],  landmarknewmouth[4+18]=1-landmarkmouth[2],landmarkmouth[18+2]
            landmarknewmouth[5],  landmarknewmouth[5+18]=1-landmarkmouth[1],landmarkmouth[18+1]
            landmarknewmouth[6],  landmarknewmouth[6+18]=1-landmarkmouth[0],landmarkmouth[18+0]
            landmarknewmouth[7],  landmarknewmouth[7+18]=1-landmarkmouth[11],landmarkmouth[18+11]
            landmarknewmouth[8],  landmarknewmouth[8+18]=1-landmarkmouth[10],landmarkmouth[18+10]
            landmarknewmouth[9],  landmarknewmouth[9+18]=1-landmarkmouth[9],landmarkmouth[18+9]
            landmarknewmouth[10],landmarknewmouth[10+18]=1-landmarkmouth[8],landmarkmouth[18+8]
            landmarknewmouth[11],landmarknewmouth[11+18]=1-landmarkmouth[7],landmarkmouth[18+7]
            landmarknewmouth[12],landmarknewmouth[12+18]=1-landmarkmouth[14],landmarkmouth[18+14]
            landmarknewmouth[13],landmarknewmouth[13+18]=1-landmarkmouth[13],landmarkmouth[18+13]
            landmarknewmouth[14],landmarknewmouth[14+18]=1-landmarkmouth[12],landmarkmouth[18+12]
            landmarknewmouth[15],landmarknewmouth[15+18]=1-landmarkmouth[17],landmarkmouth[18+17]
            landmarknewmouth[16],landmarknewmouth[16+18]=1-landmarkmouth[16],landmarkmouth[18+16]
            landmarknewmouth[17],landmarknewmouth[17+18]=1-landmarkmouth[15],landmarkmouth[18+15]

            landmarknewpupil[0],landmarknewpupil[0+2]=1-landmarkpupil[1],landmarkpupil[1+2]
            landmarknewpupil[1],landmarknewpupil[1+2]=1-landmarkpupil[0],landmarkpupil[0+2]


            #landmarknew=np.array(landmarknew)

            '''
            img_copy=image.copy()
            wid,hei=img_copy.size
            draw = ImageDraw.Draw(img_copy)

            for i in range(0,10):
                dx=float(landmarknewleye[i])*wid
                dy=float(landmarknewleye[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')

            for i in range(0,10):
                dx=float(landmarknewreye[i])*wid
                dy=float(landmarknewreye[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(8):
                dx=float(landmarknewlbrow[i])*wid
                dy=float(landmarknewlbrow[i+8])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(8):
                dx=float(landmarknewrbrow[i])*wid
                dy=float(landmarknewrbrow[i+8])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(10):
                dx=float(landmarknewnose[i])*wid
                dy=float(landmarknewnose[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(18):
                dx=float(landmarknewmouth[i])*wid
                dy=float(landmarknewmouth[i+18])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')


            img_copy.show()
            '''
            #raw_input('pause')

            #print(bbox)

            return {'image':image,'target':target, 'landmarkleye':landmarknewleye,'landmarkreye':landmarknewreye,'landmarklbrow':landmarknewlbrow,'landmarkrbrow':landmarknewrbrow,'landmarknose':landmarknewnose,'landmarkmouth':landmarknewmouth,'landmarkpupil':landmarknewpupil, 'path':path}
        #return {'image':image,'target':target, 'landmark':landmark, 'path':path}
        return {'image': image, 'target':target, 'landmarkleye':landmarkleye,'landmarkreye':landmarkreye,'landmarklbrow':landmarklbrow,'landmarkrbrow':landmarkrbrow,'landmarknose':landmarknose,'landmarkmouth':landmarkmouth,'landmarkpupil':landmarkpupil,'path': path}



    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation(object):
    """Rotate the image by angle.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center


    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, smaple):
        image, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth, path = sample['image'],sample['target'],sample['landmarkleye'], sample['landmarkreye'],sample['landmarklbrow'], sample['landmarkrbrow'],sample['landmarknose'], sample['landmarkmouth'],sample['path']

        """
            img (PIL Image): Image to be rotated.

        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        image=F.rotate(image, angle, self.resample, self.expand, self.center)
        img_copy=image.copy()
        img_copy.show()
        return {'image':image,'target':target, 'landmarkleye':landmarknewleye,'landmarkreye':landmarknewreye,'landmarklbrow':landmarknewlbrow,'landmarkrbrow':landmarknewrbrow,'landmarknose':landmarknewnose,'landmarkmouth':landmarknewmouth, 'path':path}

    '''
    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
    '''


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        image, target, landmark = sample['image'],sample['target'],sample['landmark']
        image = image.resize((26,26))
        h, w = image.size

        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image.crop((left,top, new_w+left,new_h+top))
        '''


        for i in range(5):
            landmark[2*i]   = landmark[2*i]-left
            landmark[2*i+1] = landmark[2*i+1]-top
        landmark[7] +=1
        landmark[9] +=1
        for i in range(5):
            landmark[2*i]   /= new_w
            landmark[2*i+1] /= new_h

        bbox[0] = (bbox[0]-left)/new_w
        bbox[1] = (bbox[1]-top)/new_h
        bbox[2] = (bbox[2]-left-new_w)/new_w
        bbox[3] = (bbox[3]-top-new_h)/new_h

        return {'image':image,'target':target,'bbox':bbox,'landmark':landmark}
        '''
        landmark[0] = (landmark[0]*w-left)/new_w
        landmark[1] = (landmark[1]*h-top)/new_h
        return {'image':image,'target':target, 'landmark':landmark}

class RandomCropPos1(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        image, target, box, landmark, path = sample['image'],sample['target'],sample['bbox'], sample['landmark'],sample['path']
        if target== 1:
            width,height = image.size
            #height,width = image.size
            #print('cropfunction:',image.size)
            #raw_input('pause')
            #x1,y1,x2,y2=bbox
            #x,y=box
            x1,y1=0.25,0.25
            x2,y2=0.75,0.75
            #print(x1,y1,x2,y2)
            #raw_input('pause')
            #  bounding box width and height
            w=(x2-x1)*width+1
            h=(y2-y1)*height+1
            #print(w,h)
            x1=x1*width
            y1=y1*height
            x2=x2*width
            y2=y2*height

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)
            #print(bbox)
            for k in range(1000):
                bbox_size = npr.randint(int(max(w, h) * 0.6), np.ceil(1.4 * max(w, h)))
                #print'bbox_size:',bbox_size
                delta_x = npr.randint(-h * 0.5, h * 0.5)
                delta_y = npr.randint(-h * 0.5, h * 0.5)
                nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size-1
                ny2 = ny1 + bbox_size-1
                #print(delta_x,delta_y)
                if nx2 >= width or ny2 >= height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                z1=x1 + w / 2 - h / 2
                z2=x1 + w / 2 + h / 2
                bbox2=[z1,y1,z2,y2]
                bbox2=np.array(bbox2)

                iou= IoU(crop_box.astype(np.float), np.expand_dims(bbox2.astype(np.float), 0))
                if iou > 0.5:
                    new_w=nx2-nx1+1
                    new_h=ny2-ny1+1
                    image = image.crop((nx1,ny1,nx2,ny2))
                    #image = image.crop((x1, y1, x2, y2))
                    #image.show()
                    #raw_input('pause')
                    #bbox[0] = round((bbox[0]-nx1)/new_w,5)
                    #bbox[1] = round((bbox[1]-ny1)/new_h,5)
                    #bbox[2] = round((bbox[2]-nx1)/new_w,5)
                    #bbox[3] = round((bbox[3]-ny1)/new_h,5)
                    #newx,newy=bbox[2]-bbox[0],bbox[3]-bbox[1]
                    #print(newx,newy)
                    newx = round((0.5*width-nx1)/new_w,5)
                    newy = round((0.5*height-ny1)/new_h,5)

                    box=[newx,newy]
                    box=np.array(box)

                    return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
            return {'image':image,'target':2, 'bbox':box, 'landmark':landmark,'path':path}
        elif target== 0 :
            return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
        elif target == -1:
            return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}


class RandomCropPos2(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        image, target, box, landmark, path = sample['image'],sample['target'],sample['bbox'],sample['landmark'],sample['path']
        if target== -1:
            width,height = image.size
            #height,width = image.size
            #print('cropfunction:',image.size)
            #raw_input('pause')
            #x1,y1,x2,y2=bbox
            #x,y=box
            #print(landmark)
            landmarkx=landmark[:9]
            landmarky=landmark[9:18]
            #print(landmarkx)
            #print(landmarky)
            #raw_input('pause')
            x1=np.amin(landmarkx)*width
            x2=np.amax(landmarkx)*width
            y1=np.amin(landmarky)*height
            y2=np.amax(landmarky)*height

            w=x2-x1+1
            h=y2-y1+1

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)
            for k in range(1000):
                bbox_size = npr.randint(int(max(w, h) * 0.6), np.ceil(1.4 * max(w, h)))
                #print'bbox_size:',bbox_size
                delta_x = npr.randint(-w * 0.5, w * 0.5)
                delta_y = npr.randint(-w * 0.5, w * 0.5)
                nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size-1
                ny2 = ny1 + bbox_size-1
                #print(delta_x,delta_y)
                if nx2 >= width or ny2 >= height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #z1=x1 + w / 2 - h / 2
                #z2=x1 + w / 2 + h / 2
                z1=y1 + h / 2 - w / 2
                z2=y1 + h / 2 + w / 2

                bbox2=[x1,z1,x2,z2]
                bbox2=np.array(bbox2)

                iou= IoU(crop_box.astype(np.float), np.expand_dims(bbox2.astype(np.float), 0))
                if iou > 0.3:
                    new_w=nx2-nx1+1
                    new_h=ny2-ny1+1
                    image = image.crop((nx1,ny1,nx2,ny2))
                    #image = image.crop((x1, y1, x2, y2))

                    #image.show()
                    #raw_input('pause')
                    #bbox[0] = round((bbox[0]-nx1)/new_w,5)
                    #bbox[1] = round((bbox[1]-ny1)/new_h,5)
                    #bbox[2] = round((bbox[2]-nx1)/new_w,5)
                    #bbox[3] = round((bbox[3]-ny1)/new_h,5)
                    #newx,newy=bbox[2]-bbox[0],bbox[3]-bbox[1]
                    #print(newx,newy)
                    #newx = round((0.5*width-nx1)/new_w,5)
                    #newy = round((0.5*height-ny1)/new_h,5)
                    for i in range(9):
                        landmark[i]=round((landmark[i]*width-nx1)/new_w,5)
                        landmark[i+9]=round((landmark[i+9]*height-ny1)/new_h,5)
                    '''
                    img_copy=image.copy()
                    wid,hei=img_copy.size
                    draw = ImageDraw.Draw(img_copy)
                    for i in range(9):
                        dx=float(landmark[i])*wid
                        dy=float(landmark[i+9])*hei
                        draw.ellipse([
                            (dx - 1.0, dy - 1.0),
                            (dx + 1.0, dy + 1.0)
                        ], outline='black')
                    img_copy.show()
                    raw_input('pause')
                    '''

                    return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
            return {'image':image,'target':2, 'bbox':box, 'landmark':landmark, 'path':path}
        elif target==0 or target==1:
            return sample


class RandomCropNeg(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        #if len(sample)==2:
            #return sample
        #image, target = sample['image'], sample['target']
        #image, target, bbox, path =sample['image'],sample['target'],sample['bbox'],sample['path']
        image, target, box, landmark, path = sample['image'],sample['target'],sample['bbox'],sample['landmark'],sample['path']
        #image, target, path =sample['image'],sample['target'],sample['path']
        #print(target)
        w, h = image.size
        if target== 0 and w>32:

            #image = image.resize((26,26))
            #w, h = image.size
            #print(h,w)
            xbox_size = npr.randint((w * 0.8), np.ceil(1.0 * w))
            #ybox_size = npr.randint((h * 0.6), np.ceil(0.9 * h))
            ybox_size = min(h-1,xbox_size)
            #print(h-ybox_size)
            left = npr.randint(0, w-xbox_size)
            top = npr.randint(0, h-ybox_size)
            image = image.crop((left,top, xbox_size+left-1,ybox_size+top-1))
            #image.show()
            #raw_input('pause')
            return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
            #return {'image':image,'target':target, 'bbox':bbox, 'path':path}
        else:
            return sample

'''
class RandomCropPos3(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        image, target, box, landmark, path = sample['image'],sample['target'],sample['bbox'],sample['landmark'],sample['path']
        if target== -1:
            width,height = image.size
            #height,width = image.size
            #print('cropfunction:',image.size)
            #raw_input('pause')
            #x1,y1,x2,y2=bbox
            #x,y=box
            #print(landmark)
            landmarkx=[]
            landmarky=[]

            for i in range(38):
                landmarkx.append(landmark[2*i])
                landmarky.append(landmark[2*i+1])


            #print(landmarkx)
            #print(landmarky)
            #raw_input('pause')
            x1=np.amin(landmarkx)
            x2=np.amax(landmarkx)
            y1=np.amin(landmarky)
            y2=np.amax(landmarky)

            w=x2-x1+1
            h=y2-y1+1

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)
            #print(bbox)

            for k in range(1000):
                bbox_size = npr.randint(int(max(w, h) * 0.6), np.ceil(1.4 * max(w, h)))
                bbox_sizeY = bbox_size*48/64
                #print'bbox_size:',bbox_size
                delta_x = npr.randint(-w * 0.5, w * 0.5)
                delta_y = npr.randint(-w * 0.5, w * 0.5)
                nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + h / 2 - bbox_sizeY / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size-1
                ny2 = ny1 + bbox_sizeY-1
                #print(delta_x,delta_y)
                if nx2 >= width or ny2 >= height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                z1=x1 + w / 2 - h / 2
                z2=x1 + w / 2 + h / 2
                #z1=y1 + h / 2 - w / 2
                #z2=y1 + h / 2 + w / 2

                bbox2=[z1,y1,z2,y2]
                bbox2=np.array(bbox2)


                iou= IoU(crop_box.astype(np.float), np.expand_dims(bbox2.astype(np.float), 0))
                iou2= IoU2(crop_box.astype(np.float), np.expand_dims(bbox.astype(np.float), 0))
                #print(iou)
                #print(iou2)
                if iou > 0.3 and iou2>0.9:
                    new_w=nx2-nx1+1
                    new_h=ny2-ny1+1
                    image = image.crop((nx1,ny1,nx2,ny2))
                    #image = image.crop((x1, y1, x2, y2))
                    #print('IOU')
                    #image.show()
                    #raw_input('pause')
                    #bbox[0] = round((bbox[0]-nx1)/new_w,5)
                    #bbox[1] = round((bbox[1]-ny1)/new_h,5)
                    #bbox[2] = round((bbox[2]-nx1)/new_w,5)
                    #bbox[3] = round((bbox[3]-ny1)/new_h,5)
                    #newx,newy=bbox[2]-bbox[0],bbox[3]-bbox[1]
                    #print(newx,newy)
                    #newx = round((0.5*width-nx1)/new_w,5)
                    #newy = round((0.5*height-ny1)/new_h,5)

                    for i in range(18):
                        landmarkx[i]=round((landmarkx[i]-nx1)/new_w,5)
                        landmarky[i]=round((landmarky[i]-ny1)/new_h,5)


                    landmark=np.array(landmarkx[0:18]+landmarky[0:18])

                    img_copy=image.copy()
                    wid,hei=img_copy.size
                    draw = ImageDraw.Draw(img_copy)
                    for i in range(18):
                        dx=float(landmark[i])*wid
                        dy=float(landmark[i+18])*hei
                        draw.ellipse([
                            (dx - 1.0, dy - 1.0),
                            (dx + 1.0, dy + 1.0)
                        ], outline='black')
                    img_copy.show()
                    #raw_input('pause')


                    return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
            return {'image':image,'target':2, 'bbox':box, 'landmark':landmark[0:36], 'path':path}
        elif target==0 or target==1:
            return sample
'''

class RandomCropPos3(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        #image, target, box, landmarkeye, landmarkbrow, path = sample['image'],sample['target'],sample['bbox'],sample['landmarkeye'],sample['landmarkbrow'],sample['path']
        #if target== -1:
        #image, target, landmark, path = sample['image'],sample['target'],sample['landmark'],sample['path']
        image, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth,landmarkpupil, path = sample['image'],sample['target'],sample['landmarkleye'], sample['landmarkreye'],sample['landmarklbrow'], sample['landmarkrbrow'],sample['landmarknose'], sample['landmarkmouth'], sample['landmarkpupil'],sample['path']
        if target== 0 or target== 1 or target==-1:
        #if  target==-1:
            width,height = image.size

            landmarkx=[]
            landmarky=[]

            for i in range(10):
                landmarkx.append(landmarkleye[2*i])
                landmarky.append(landmarkleye[2*i+1])
            for i in range(10):
                landmarkx.append(landmarkreye[2*i])
                landmarky.append(landmarkreye[2*i+1])
            for i in range(8):
                landmarkx.append(landmarklbrow[2*i])
                landmarky.append(landmarklbrow[2*i+1])
            for i in range(8):
                landmarkx.append(landmarkrbrow[2*i])
                landmarky.append(landmarkrbrow[2*i+1])
            for i in range(10):
                landmarkx.append(landmarknose[2*i])
                landmarky.append(landmarknose[2*i+1])
            for i in range(18):
                landmarkx.append(landmarkmouth[2*i])
                landmarky.append(landmarkmouth[2*i+1])
            for i in range(2):
                landmarkx.append(landmarkpupil[2*i])
                landmarky.append(landmarkpupil[2*i+1])


            x1=np.amin(landmarkx)
            x2=np.amax(landmarkx)
            y2=np.amax(landmarky)
            y1=np.amin(landmarky)

            h=y2-y1+1
            y2=y2+h*0.15
            y1=max(y1-h*0.15,0)
            w=x2-x1+1
            h=y2-y1+1

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)

            for k in range(3000):
                bbox_size = npr.randint(int(max(w, h) * 0.9), np.ceil(1.5 * max(w, h)))
                #bbox_size = npr.randint(int(max(w, h) * 0.8), np.ceil(2.0 * max(w, h)))
                #bbox_sizeY = bbox_size*48/64
                bbox_sizeY = bbox_size
                bbox_sizeY = npr.randint(int(bbox_sizeY * 0.9), np.ceil(1.1 * bbox_sizeY))
                #print'bbox_size:',bbox_size
                delta_x = npr.randint(-w * 0.35, w * 0.35)
                delta_y = npr.randint(-w * 0.35, w * 0.15)
                nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + h / 2 - bbox_sizeY / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size-1
                ny2 = ny1 + bbox_sizeY-1
                #print(delta_x,delta_y)
                if nx2 >= width or ny2 >= height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])


                lbwidth=abs(landmarkx[24]-landmarkx[20])+abs(landmarky[24]-landmarky[20])*0.5
                rbwidth=abs(landmarkx[32]-landmarkx[28])+abs(landmarky[32]-landmarky[28])*0.5

                weight_h=min(max(float(lbwidth)/float(rbwidth+lbwidth),0.2),0.8)

                z1=x1 + w / 2 - h *weight_h
                z2=x1 + w / 2 + h *(1.0-weight_h)

                bbox2=[z1,y1,z2,y2]
                bbox2=np.array(bbox2)

                iou= IoU(crop_box.astype(np.float), np.expand_dims(bbox2.astype(np.float), 0))
                iou2= IoU2(crop_box.astype(np.float), np.expand_dims(bbox.astype(np.float), 0))

                if iou > 0.5 and iou2>=0.9:
                #if iou > 0.2:
                    image = image.crop((nx1,ny1,nx2,ny2))
                    new_w=nx2-nx1+1
                    new_h=ny2-ny1+1

                    for i in range(66):
                        landmarkx[i]=round((landmarkx[i]-nx1)/new_w,5)
                        landmarky[i]=round((landmarky[i]-ny1)/new_h,5)


                    #landmarkCrop=np.array(landmarkx[0:64]+landmarky[0:64])
                    landmarkCropleye=np.array(landmarkx[0:10]+landmarky[0:10])
                    landmarkCropreye=np.array(landmarkx[10:20]+landmarky[10:20])
                    landmarkCroplbrow=np.array(landmarkx[20:28]+landmarky[20:28])
                    landmarkCroprbrow=np.array(landmarkx[28:36]+landmarky[28:36])
                    landmarkCropnose=np.array(landmarkx[36:46]+landmarky[36:46])
                    landmarkCropmouth=np.array(landmarkx[46:64]+landmarky[46:64])
                    landmarkCroppupil=np.array(landmarkx[64:66]+landmarky[64:66])

                    #image.show()
                    '''
                    img_copy=image.copy()
                    wid,hei=img_copy.size
                    draw = ImageDraw.Draw(img_copy)

                    for i in range(0,10):
                        dx=float(landmarkCropleye[i])*wid
                        dy=float(landmarkCropleye[i+10])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')

                    for i in range(0,10):
                        dx=float(landmarkCropreye[i])*wid
                        dy=float(landmarkCropreye[i+10])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')
                    for i in range(8):
                        dx=float(landmarkCroplbrow[i])*wid
                        dy=float(landmarkCroplbrow[i+8])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')
                    for i in range(8):
                        dx=float(landmarkCroprbrow[i])*wid
                        dy=float(landmarkCroprbrow[i+8])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')
                    for i in range(10):
                        dx=float(landmarkCropnose[i])*wid
                        dy=float(landmarkCropnose[i+10])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')
                    for i in range(18):
                        dx=float(landmarkCropmouth[i])*wid
                        dy=float(landmarkCropmouth[i+18])*hei
                        if i%2==0:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='black')
                        else:
                            draw.ellipse([
                                (dx - 1.0, dy - 1.0),
                                (dx + 1.0, dy + 1.0)
                            ], outline='white')


                    img_copy.show()
                    '''
                    return {'image': image, 'target':target, 'landmarkleye':landmarkCropleye,'landmarkreye':landmarkCropreye,'landmarklbrow':landmarkCroplbrow,'landmarkrbrow':landmarkCroprbrow,'landmarknose':landmarkCropnose,'landmarkmouth':landmarkCropmouth,'landmarkpupil':landmarkCroppupil,'path': path}
                    #return {'image':image,'target':target, 'landmark':landmarkCrop, 'path':path}
            #return {'image':image,'target':2, 'landmark':landmark, 'path':path}
            #img_copy=image.copy()
            #img_copy.show()
            return {'image': image, 'target':2, 'landmarkleye':landmarkleye,'landmarkreye':landmarkreye,'landmarklbrow':landmarklbrow,'landmarkrbrow':landmarkrbrow,'landmarknose':landmarknose,'landmarkmouth':landmarkmouth,'landmarkpupil':landmarkpupil,'path': path}
            #return {'image': image, 'target':-1, 'landmarkleye':landmarkleye,'landmarkreye':landmarkreye,'landmarklbrow':landmarklbrow,'landmarkrbrow':landmarkrbrow,'landmarknose':landmarknose,'landmarkmouth':landmarkmouth,'path': path}
        elif target==0 or target==1:
            return sample


class RandomCropPos4(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        image, target, box, landmark, path = sample['image'],sample['target'],sample['bbox'], sample['landmark'],sample['path']

        if target== 1:
            width,height = image.size
            lx,ly,rx,ry=box
            #print(width,height)
            #print(lx,ly,rx,ry)
            #raw_input('pause')
            xmin,xmax=min(lx,rx),max(lx,rx)
            ymin,ymax=min(ly,ry),max(ly,ry)
            x1=min(max(lx-width/8,0),width)
            y1=min(max(ly-height/8,0),height)
            x2=min(max(rx+width/8,0),width)
            y2=min(max(ry+height/8,0),height)

            w=x2-x1+1
            h=y2-y1+1

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)
            #print(bbox)
            for k in range(1000):
                bbox_size = npr.randint(int(max(w, h) * 0.6), np.ceil(1.4 * max(w, h)))
                bbox_sizeY = bbox_size*48/64
                #print'bbox_size:',bbox_size
                delta_x = npr.randint(-h * 0.5, h * 0.5)
                delta_y = npr.randint(-h * 0.5, h * 0.5)
                nx1 = max(x1 + w / 2 - bbox_size / 2 + delta_x, 0)
                ny1 = max(y1 + h / 2 - bbox_size / 2 + delta_y, 0)

                nx2 = nx1 + bbox_size-1
                ny2 = ny1 + bbox_sizeY-1
                #print(delta_x,delta_y)
                if nx2 >= width or ny2 >= height:
                    continue
                crop_box = np.array([nx1, ny1, nx2, ny2])
                #z1=x1 + w / 2 - h / 2
                #z2=x1 + w / 2 + h / 2
                #bbox2=[z1,y1,z2,y2]
                z1= y1+h/2-w/2
                z2= y1+h/2+w/2
                bbox2=[x1,z1,x2,z2]

                bbox2=np.array(bbox2)

                iou= IoU(crop_box.astype(np.float), np.expand_dims(bbox2.astype(np.float), 0))
                #iou2= IoU(crop_box.astype(np.float), np.expand_dims(bbox.astype(np.float), 0))
                #if iou > 0.3 and iou2>0.7:
                if iou > 0.7 :
                    new_w=nx2-nx1+1
                    new_h=ny2-ny1+1
                    image = image.crop((nx1,ny1,nx2,ny2))
                    #image = image.crop((x1, y1, x2, y2))
                    #image.show()
                    #raw_input('pause')
                    newlx = round((lx-nx1)/new_w,5)
                    newly = round((ly-ny1)/new_h,5)
                    newrx = round((rx-nx1)/new_w,5)
                    newry = round((ry-ny1)/new_h,5)

                    box=[newlx,newly,newrx,newry]
                    box=np.array(box)

                    '''
                    img_copy=image.copy()
                    wid,hei=img_copy.size
                    draw = ImageDraw.Draw(img_copy)
                    for i in range(2):
                        dx=float(box[i*2])*wid
                        dy=float(box[i*2+1])*hei
                        draw.ellipse([
                            (dx - 1.0, dy - 1.0),
                            (dx + 1.0, dy + 1.0)
                        ], outline='black')
                    img_copy.show()
                    #raw_input('pause')
                    '''

                    return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
            return {'image':image,'target':2, 'bbox':box, 'landmark':landmark,'path':path}
        elif target== 0 :
            return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}
        elif target == -1:
            return {'image':image,'target':target, 'bbox':box, 'landmark':landmark, 'path':path}




class testcrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        if len(sample)==2:
            return sample
        #image, target, box, landmarkeye, landmarkbrow, path = sample['image'],sample['target'],sample['bbox'],sample['landmarkeye'],sample['landmarkbrow'],sample['path']
        #if target== -1:
        #image, target, landmark, path = sample['image'],sample['target'],sample['landmark'],sample['path']
        image, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth, path = sample['image'],sample['target'],sample['landmarkleye'], sample['landmarkreye'],sample['landmarklbrow'], sample['landmarkrbrow'],sample['landmarknose'], sample['landmarkmouth'],sample['path']
        if target== 0 or target== 1 or target==-1:
        #if  target==-1:
            width,height = image.size

            landmarkx=[]
            landmarky=[]

            for i in range(10):
                landmarkx.append(landmarkleye[2*i])
                landmarky.append(landmarkleye[2*i+1])
            for i in range(10):
                landmarkx.append(landmarkreye[2*i])
                landmarky.append(landmarkreye[2*i+1])
            for i in range(8):
                landmarkx.append(landmarklbrow[2*i])
                landmarky.append(landmarklbrow[2*i+1])
            for i in range(8):
                landmarkx.append(landmarkrbrow[2*i])
                landmarky.append(landmarkrbrow[2*i+1])
            for i in range(10):
                landmarkx.append(landmarknose[2*i])
                landmarky.append(landmarknose[2*i+1])
            for i in range(18):
                landmarkx.append(landmarkmouth[2*i])
                landmarky.append(landmarkmouth[2*i+1])

            x1=np.amin(landmarkx)
            x2=np.amax(landmarkx)
            y2=np.amax(landmarky)
            y1=np.amin(landmarky)


            w=x2-x1+1
            h=y2-y1+1

            if w>=h:
                y1=max(y1-(w-h)/2,0)
                y2=min(y2+(w-h)/2,height)
                h=w
            else:
                x1=max(x1-(h-w)/2,0)
                x2=min(x2+(h-w)/2,width)
                w=h

            bbox=[x1,y1,x2,y2]
            bbox=np.array(bbox)
            for i in range(64):
                landmarkx[i]=round((landmarkx[i]-x1)/w,5)
                landmarky[i]=round((landmarky[i]-y1)/h,5)
            landmarkCropleye=np.array(landmarkx[0:10]+landmarky[0:10])
            landmarkCropreye=np.array(landmarkx[10:20]+landmarky[10:20])
            landmarkCroplbrow=np.array(landmarkx[20:28]+landmarky[20:28])
            landmarkCroprbrow=np.array(landmarkx[28:36]+landmarky[28:36])
            landmarkCropnose=np.array(landmarkx[36:46]+landmarky[36:46])
            landmarkCropmouth=np.array(landmarkx[46:64]+landmarky[46:64])
            image = image.crop((x1,y1,x2,y2))
            '''
            img_copy=image.copy()
            wid,hei=img_copy.size
            print(wid,hei)
            draw = ImageDraw.Draw(img_copy)

            for i in range(0,10):
                dx=float(landmarkCropleye[i])*wid
                dy=float(landmarkCropleye[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')

            for i in range(0,10):
                dx=float(landmarkCropreye[i])*wid
                dy=float(landmarkCropreye[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(8):
                dx=float(landmarkCroplbrow[i])*wid
                dy=float(landmarkCroplbrow[i+8])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(8):
                dx=float(landmarkCroprbrow[i])*wid
                dy=float(landmarkCroprbrow[i+8])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(10):
                dx=float(landmarkCropnose[i])*wid
                dy=float(landmarkCropnose[i+10])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')
            for i in range(18):
                dx=float(landmarkCropmouth[i])*wid
                dy=float(landmarkCropmouth[i+18])*hei
                if i%2==0:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='black')
                else:
                    draw.ellipse([
                        (dx - 1.0, dy - 1.0),
                        (dx + 1.0, dy + 1.0)
                    ], outline='white')


            img_copy.show()
            '''
        return {'image': image, 'target':target, 'landmarkleye':landmarkCropleye,'landmarkreye':landmarkCropreye,'landmarklbrow':landmarkCroplbrow,'landmarkrbrow':landmarkCroprbrow,'landmarknose':landmarkCropnose,'landmarkmouth':landmarkCropmouth,'path': path}


def default_loader(path):
    img = Image.open(path).convert('L')
    #print('image:',img.size)
    return img

def default_list_reader(fileList):
    imgList = []
    num=0
    with open(fileList, 'r') as file:
        for line in file.readlines():
            '''
            cls: 1 positive 0 negtive
            BB: only for 1 positve
            '''
            i= line.strip().split(' ')
            imgPath= i[0]
            label= i[1]
            # num+=1
            # if num%6>0:
            #     continue;
            #print ('label:',label)

            
            landmarkleye= map(float,i[2:22])
            landmarkleye= np.asarray(landmarkleye,dtype='float32')
            landmarkreye= map(float,i[22:42])
            landmarkreye= np.asarray(landmarkreye,dtype='float32')
            landmarklbrow= map(float,i[42:58])
            landmarklbrow= np.asarray(landmarklbrow,dtype='float32')
            landmarkrbrow= map(float,i[58:74])
            landmarkrbrow= np.asarray(landmarkrbrow,dtype='float32')
            landmarknose= map(float,i[74:94])
            landmarknose= np.asarray(landmarknose,dtype='float32')
            landmarkmouth= map(float,i[94:130])
            landmarkmouth= np.asarray(landmarkmouth,dtype='float32')
            landmarkpupil= map(float,i[130:134])
            landmarkpupil= np.asarray(landmarkpupil,dtype='float32')


            '''
            else:
                landmark= np.ones([128],dtype='float32')*0.5
            '''
            imgList.append((imgPath, int(label),landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth,landmarkpupil))
            #imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, fileList,custom_transform=None, image_transform=None, list_reader= default_list_reader, loader= default_loader, root=None):
        #self.root      = root
        self.imgList   = list_reader(fileList)
        self.custom_transform = custom_transform
        self.image_transform = image_transform
        self.loader = loader
        self.root = root
    def __getitem__(self, index):
        imgPath, target,landmarkleye,landmarkreye,landmarklbrow,landmarkrbrow,landmarknose,landmarkmouth,landmarkpupil = self.imgList[index]

        landmarkleye= landmarkleye.astype('float')
        landmarkreye= landmarkreye.astype('float')
        landmarklbrow= landmarklbrow.astype('float')
        landmarkrbrow= landmarkrbrow.astype('float')
        landmarknose= landmarknose.astype('float')
        landmarkmouth= landmarkmouth.astype('float')
        landmarkpupil= landmarkpupil.astype('float')
        img = self.loader(os.path.join(self.root,imgPath))

        #sample={'image': img, 'target':target, 'bbox':bbox , 'landmarkeye':landmarkeye, 'landmarkbrow':landmarkbrow ,'path': imgPath}
        sample={'image': img, 'target':target, 'landmarkleye':landmarkleye,'landmarkreye':landmarkreye,
                                                'landmarklbrow':landmarklbrow,'landmarkrbrow':landmarkrbrow,
                                                'landmarknose':landmarknose,'landmarkmouth':landmarkmouth,
                                                'landmarkpupil':landmarkpupil,'path': imgPath}
        if self.custom_transform is not None:
            sample = self.custom_transform(sample)
        if self.image_transform is not None:
            sample['image']= self.image_transform(sample['image'])
        #return sample['image'], sample['target']
        return sample['image'], sample['target'], sample['landmarkleye'], sample['landmarkreye'], sample['landmarklbrow'], sample['landmarkrbrow'], sample['landmarknose'], sample['landmarkmouth'], sample['landmarkpupil']

    def __len__(self):
        return len(self.imgList)



if __name__=='__main__':

    data_transforms = {
        'custom': transforms.Compose([
            #testcrop((128,128)),
            RandomCropPos3((32,32)),
            RandomHorizontalFlip(),
            Randombrighten(),

        ]),
        'image': transforms.Compose([
            transforms.Resize((256,256)),
            #transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    }
    data_dir = ''
    filelist='/home/liufei/Documents/data/clpupil66.txt'
    dataset= ImageList(fileList= filelist, custom_transform= data_transforms['custom'], image_transform= data_transforms['image'],root=data_dir)
    #print(len(dataset))
    #print(dataset[0])

    train_loader = data.DataLoader(dataset,batch_size=3, shuffle=True ,num_workers=1, pin_memory=True)
    print(len(train_loader))
    
    for i in enumerate(train_loader):
        #print(i[1])
        #print(i[1][0].size())#img
        print(i[1][1])#label
        #print(i[1][2])#lmleye
        #print(i[1][3])#lmreye
        #print(i[1][4])#lmlbrow
        #print(i[1][5])#lmrbrow
        #print(i[1][6])#lmnose
        #print(i[1][7])#lmmouth
        print(i[1][8])#lmpupil
        raw_input('pause')