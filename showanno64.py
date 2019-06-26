import os
import cv2

#datadir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/VGG500_hdpose'
datadir='/home/ficha/Documents/riku/pupil/testpic'


for j in os.listdir(datadir):
    picfile=datadir+'/'+j
    if '.png' in picfile:
        #print(j)
        annofile=picfile[:-4]+'.txt'
        #raw_input('pause')
        img= cv2.imread(picfile)
        print(img.shape)
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        with open(annofile) as f:
            lines=f.readlines()
            landmark=[]
            for m in lines:
                points=m.strip('\n').split(' ')
                #print(points)
                #x=int(points[0])
                #y=int(points[1])
                point=(int(float(points[0])),int(float(points[1])))
                print(point)

                landmark.append(point[0])
                landmark.append(point[1])
            #print(len(landmark))
            #print(landmark)
            #raw_input('pause')

            for i in range(38):
                l=landmark[2*i]
                r=landmark[2*i+1]
                #print('l:',l)
                #print('r:',r)
                #raw_input('pause')
                cv2.circle(img,(l,r),2,(255,0,0),-1)
            #x1,y1,x2,y2=bbox
            #cv2.rectangle(img,(x1,y1),(x1+x2,y1+y2),(0,255,0),1)
            cv2.imshow('img',img)
            cv2.waitKey(0)
