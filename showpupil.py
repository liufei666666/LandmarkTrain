import os
import cv2

def merge_pupil(file):
    for j in os.listdir(file):
        person=file+'/'+j
        if os.path.isdir(person):
            #print(person)
            #raw_input('pause')
            for i in os.listdir(person):
                if '.csv' in i:
                    #print(i)
                    csvfile=person+'/'+i
                    #print(csvfile)
                    #raw_input('pause')
                    with open(csvfile,'r') as t:
                        lines=t.readlines()
                        for line in lines:
                            line=line.strip('\n').split(' ')
                            #print(line)
                            #raw_input()
                            picname=person+'/'+ line[0]
                            picanno= picname[:-4]+'_fplus.txt'
                            sign=line[7]    
                            lpupilx,lpupily=line[3],line[4]
                            rpupilx,rpupily=line[10],line[11]
                            #if sign=='1':
                                #print(picname)
                                #print(picanno)
                                #raw_input('pause')
                                # with open(picanno,'a') as  pa:
                                #     pa.write('pupil:\n')
                                #     pa.write(lpupilx+' '+lpupily+'\n')
                                #     pa.write(rpupilx+' '+rpupily+'\n')


def get_fplus_build66(persondir,txtfile):

    for j in os.listdir(persondir):
        #print(j)
        #print(persondir)
        annofile=persondir+'/'+j

        if '_fplus.txt' in annofile:
        #if '.txt' in annofile:
            #print(annofile)
            #raw_input('pause')

            picfile=annofile[:-10]+'.png'
            #picfile=annofile[:-4]+'.png'
            if not os.path.exists(annofile):
                continue
            #raw_input('pause')
            #img= cv2.imread(picfile)
            #print(img.shape)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)

            with open(annofile) as f:
                #print(annofile)
                #raw_input('pause')
                #fplus
                lines=f.readlines()[1:]
                #print(len(lines))
                #raw_input('pause')
                leyeline=lines[0:10]
                reyeline=lines[10:20]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                if len(lines)==67:
                    pupilline= lines[65:67]
                    #print(pupilline)
                    label='1'
                elif len(lines)==2:
                    pupilline= lines[:2]
                    lines=['0 0\n']*64
                    leyeline=lines[0:10]
                    reyeline=lines[10:20]
                    lbrowline=lines[20:28]
                    rbrowline=lines[28:36]
                    noseline=lines[36:46]
                    mouthline= lines[46:64] 

                    label='2'
                elif len(lines)==64:
                    pupilline= ['0 0\n']*2
                    label='0'
                landmarks=[]

                for leye in leyeline:
                    points=leye.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for reye in reyeline:
                    points=reye.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for lbrow in lbrowline:
                    points=lbrow.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for rbrow in rbrowline:
                    points=rbrow.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for nose in noseline:
                    points= nose.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for mouth in mouthline:
                    points= mouth.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for pupil in pupilline:
                    points= pupil.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])


                #print(len(landmarks))
                #raw_input('pause')


                for i in range(66):
                    try:
                        l=landmarks[2*i]
                        r=landmarks[2*i+1]
                    except:
                        print(len(lines))
                        print(annofile)   
                        raw_input('pause')
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' '+label+' ')
                    for i in range(66):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')


def merge_list(cleanfile,txtfile):
    with open(cleanfile,'r') as cl:
        for line in cl.readlines():
            #print(line)
            line=line.strip('\n').split(' ')
            picfile=line[0]
            annofile=picfile[:-4]+'_fplus.txt' 
            with open(annofile) as f:
                #print(annofile)
                #raw_input('pause')
                #fplus
                lines=f.readlines()[1:]
                #print(len(lines))
                #raw_input('pause')
                leyeline=lines[0:10]
                reyeline=lines[10:20]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                if len(lines)==67:
                    pupilline= lines[65:67]
                    #print(pupilline)
                    label='1'
                elif len(lines)==2:
                    pupilline= lines[:2]
                    lines=['0 0\n']*64
                    leyeline=lines[0:10]
                    reyeline=lines[10:20]
                    lbrowline=lines[20:28]
                    rbrowline=lines[28:36]
                    noseline=lines[36:46]
                    mouthline= lines[46:64] 

                    label='2'
                elif len(lines)==64:
                    pupilline= ['0 0\n']*2
                    label='0'
                landmarks=[]

                for leye in leyeline:
                    points=leye.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for reye in reyeline:
                    points=reye.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for lbrow in lbrowline:
                    points=lbrow.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for rbrow in rbrowline:
                    points=rbrow.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for nose in noseline:
                    points= nose.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for mouth in mouthline:
                    points= mouth.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for pupil in pupilline:
                    points= pupil.strip('\n').split(' ')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])


                #print(len(landmarks))
                #raw_input('pause')


                for i in range(66):
                    try:
                        l=landmarks[2*i]
                        r=landmarks[2*i+1]
                    except:
                        print(len(lines))
                        print(annofile)   
                        raw_input('pause')
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' '+label+' ')
                    for i in range(66):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')
      
            #raw_input('pause') 

if __name__ == '__main__':
    '''
    file='/home/liufei/Documents/data/pupil-annotation'    
    #merge_pupil(file)
    
    txtfile='/home/liufei/Documents/data/pupil-annotation_pupil66.txt'

    for i in os.listdir(file):
        persondir=file+'/'+i
        print(persondir)
        get_fplus_build66(persondir,txtfile)
    '''
    cleanfile='/home/liufei/Documents/data/LM4_nc.txt'
    clpupilfile='/home/liufei/Documents/data/clpupil66.txt'
    merge_list(cleanfile,clpupilfile)