import os
import cv2


def get_fplus_build64(persondir,txtfile):

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
                leyeline=lines[0:10]
                reyeline=lines[10:20]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                landmarks=[]


                #leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[8]]
                #reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[8]]
                #lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                #rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                #noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]

                '''
                #LMnet&fraw to eyelist
                lines=f.readlines()
                leyeline=lines[0:10]
                lbrowline=lines[10:18]
                mouthline= lines[18:36]
                noseline=lines[36:46]
                reyeline=lines[46:56]
                rbrowline=lines[56:64]
                landmarks=[]



                leyeline=[leyeline[2],leyeline[3],leyeline[0],leyeline[4],leyeline[6],leyeline[9],leyeline[7],leyeline[8],leyeline[1],leyeline[5]]
                reyeline=[reyeline[2],reyeline[3],reyeline[0],reyeline[4],reyeline[6],reyeline[9],reyeline[7],reyeline[8],reyeline[1],reyeline[5]]
                lbrowline=[lbrowline[0],lbrowline[1],lbrowline[2],lbrowline[3],lbrowline[4],lbrowline[7],lbrowline[6],lbrowline[5]]
                rbrowline=[rbrowline[0],rbrowline[1],rbrowline[2],rbrowline[3],rbrowline[4],rbrowline[7],rbrowline[6],rbrowline[5]]
                noseline=[noseline[0],noseline[1],noseline[7],noseline[2],noseline[3],noseline[6],noseline[8],noseline[5],noseline[4],noseline[9]]
                '''
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



                #print(len(landmarks))
                #raw_input('pause')


                for i in range(64):
                    l=landmarks[2*i]
                    r=landmarks[2*i+1]
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' -1 ')
                    for i in range(64):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')



def get_fplus_build38(persondir,txtfile):

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
            img= cv2.imread(picfile)
            #print(img.shape)
            #cv2.imshow('image',img)
            #cv2.waitKey(0)

            with open(annofile) as f:
                #print(annofile)
                #raw_input('pause')
                lines=f.readlines()[1:]
                #lines=f.readlines()


                #fplus to eyelist
                leyeline=lines[0:9]
                reyeline=lines[10:19]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                landmarks=[]



                leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[9]]
                reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[9]]
                lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]
                '''

                #LMnet&fraw to eyelist
                leyeline=lines[0:9]
                lbrowline=lines[10:18]
                mouthline= lines[18:36]
                noseline=lines[36:46]
                reyeline=lines[46:55]
                rbrowline=lines[56:64]
                landmarks=[]



                leyeline=[leyeline[2],leyeline[8],leyeline[7],leyeline[9],leyeline[6],leyeline[4],leyeline[0],leyeline[3],leyeline[1]]
                reyeline=[reyeline[2],reyeline[8],reyeline[7],reyeline[9],reyeline[6],reyeline[4],reyeline[0],reyeline[3],reyeline[1]]
                lbrowline=[lbrowline[0],lbrowline[5],lbrowline[6],lbrowline[7],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                rbrowline=[rbrowline[0],rbrowline[5],rbrowline[6],rbrowline[7],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                noseline=[noseline[0],noseline[1],noseline[4],noseline[5]]
                '''


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


                #print(len(landmarks))
                #raw_input('pause')


                for i in range(38):
                    l=landmarks[2*i]
                    r=landmarks[2*i+1]
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' -1 ')
                    for i in range(38):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')


def get_fplus_desay(persondir,txtfile):

    for j in os.listdir(persondir):
        #print(j)
        #print(persondir)
        annofile=persondir+'/'+j

        #if '_fplus.txt' in annofile:
        if '.pts' in annofile:
            #print(annofile)
            #raw_input('pause')

            picfile=annofile[:-4]+'.png'
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
                #lines=f.readlines()
                #print(lines[0])
                #raw_input('pause')
                if '{' in lines[0]:
                    #print(picfile)
                    #print(lines[0])
                    lines=lines[1:]

                leyeline=lines[0:10]
                reyeline=lines[10:20]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                landmarks=[]
                #print(leyeline)
                #raw_input('pause')

                #leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[8]]
                #reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[8]]
                #lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                #rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                #noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]
                '''

                #LMnet&fraw to eyelist
                lines=f.readlines()
                leyeline=lines[0:10]
                lbrowline=lines[10:18]
                mouthline= lines[18:36]
                noseline=lines[36:46]
                reyeline=lines[46:56]
                rbrowline=lines[56:64]
                landmarks=[]



                leyeline=[leyeline[2],leyeline[3],leyeline[0],leyeline[4],leyeline[6],leyeline[9],leyeline[7],leyeline[8],leyeline[1],leyeline[5]]
                reyeline=[reyeline[2],reyeline[3],reyeline[0],reyeline[4],reyeline[6],reyeline[9],reyeline[7],reyeline[8],reyeline[1],reyeline[5]]
                lbrowline=[lbrowline[0],lbrowline[1],lbrowline[2],lbrowline[3],lbrowline[4],lbrowline[7],lbrowline[6],lbrowline[5]]
                rbrowline=[rbrowline[0],rbrowline[1],rbrowline[2],rbrowline[3],rbrowline[4],rbrowline[7],rbrowline[6],rbrowline[5]]
                noseline=[noseline[0],noseline[1],noseline[7],noseline[2],noseline[3],noseline[6],noseline[8],noseline[5],noseline[4],noseline[9]]
                '''
                for leye in leyeline:
                    #print(leye.strip())
                    if '\t' in leye:
                        points=leye.strip('\r\n').split('\t')
                    else:
                        points=leye.strip('\r\n').split(' ')
                    #print(points)
                    #print(points)
                    #raw_input('pause')
                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for reye in reyeline:
                    if '\t' in reye:
                        points=reye.strip('\r\n').split('\t')
                    else:
                        points=reye.strip('\r\n').split(' ')

                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for lbrow in lbrowline:
                    if '\t' in lbrow:
                        points=lbrow.strip('\r\n').split('\t')
                    else:
                        points=lbrow.strip('\r\n').split(' ')

                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for rbrow in rbrowline:
                    if '\t' in rbrow:
                        points=rbrow.strip('\r\n').split('\t')
                    else:
                        points=rbrow.strip('\r\n').split(' ')

                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for nose in noseline:
                    if '\t' in nose:
                        points= nose.strip('\r\n').split('\t')
                    else:
                        points= nose.strip('\r\n').split(' ')

                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])

                for mouth in mouthline:
                    if '\t' in mouth:
                        points= mouth.strip('\r\n').split('\t')
                    else:
                        points= mouth.strip('\r\n').split(' ')

                    point=(int(float(points[0])),int(float(points[1])))
                    landmarks.append(point[0])
                    landmarks.append(point[1])



                #print(len(landmarks))
                #raw_input('pause')


                for i in range(64):
                    l=landmarks[2*i]
                    r=landmarks[2*i+1]
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' -1 ')
                    for i in range(64):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')


def get_lm11a_build64(persondir,txtfile):

    for j in os.listdir(persondir):
        #print(j)
        annofile=persondir+'/'+j
        #print(annofile)

        if '_lm11a.txt' in annofile:
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
                leyeline=lines[0:10]
                reyeline=lines[10:20]
                lbrowline=lines[20:28]
                rbrowline=lines[28:36]
                noseline=lines[36:46]
                mouthline= lines[46:64]
                landmarks=[]


                #leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[8]]
                #reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[8]]
                #lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                #rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                #noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]

                '''
                #LMnet&fraw to eyelist
                lines=f.readlines()
                leyeline=lines[0:10]
                lbrowline=lines[10:18]
                mouthline= lines[18:36]
                noseline=lines[36:46]
                reyeline=lines[46:56]
                rbrowline=lines[56:64]
                landmarks=[]



                leyeline=[leyeline[2],leyeline[3],leyeline[0],leyeline[4],leyeline[6],leyeline[9],leyeline[7],leyeline[8],leyeline[1],leyeline[5]]
                reyeline=[reyeline[2],reyeline[3],reyeline[0],reyeline[4],reyeline[6],reyeline[9],reyeline[7],reyeline[8],reyeline[1],reyeline[5]]
                lbrowline=[lbrowline[0],lbrowline[1],lbrowline[2],lbrowline[3],lbrowline[4],lbrowline[7],lbrowline[6],lbrowline[5]]
                rbrowline=[rbrowline[0],rbrowline[1],rbrowline[2],rbrowline[3],rbrowline[4],rbrowline[7],rbrowline[6],rbrowline[5]]
                noseline=[noseline[0],noseline[1],noseline[7],noseline[2],noseline[3],noseline[6],noseline[8],noseline[5],noseline[4],noseline[9]]
                '''
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



                #print(len(landmarks))
                #raw_input('pause')


                for i in range(64):
                    l=landmarks[2*i]
                    r=landmarks[2*i+1]
                    #print('l:',l)
                    #print('r:',r)
                    #raw_input('pause')
                    #cv2.circle(img,(l,r),2,(0,0,255),-1)
                #cv2.imshow('img',img)
                #cv2.waitKey(0)


                with open(txtfile,'a') as f:
                    f.write(picfile+ ' -1 ')
                    for i in range(64):
                        f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                    f.write('\n')



def get_fplus_clean_build64LM(persondir,cleantxt,txtfile):
    ff=open(cleantxt)
    clines=ff.readlines()
    newclines=[]
    count=0
    #print(clines[:10])
    for cline in clines:
        newcline= cline.strip('\n')
        newcline=os.path.basename(newcline)
        #print(newcline)
        #raw_input('pause')
        newclines.append(newcline)
    print('num of clean data:',len(newclines))
    ff.close()
    for j in os.listdir(persondir):
        #print(j)
        #print(persondir)
        annofile=persondir+'/'+j

        if '_fplus.txt' in annofile:
            #print(annofile)

            picfile=annofile[:-10]+'.png'
            picbsname=os.path.basename(picfile)
            #print(picbsname)
            #raw_input('pause')
            if picbsname in newclines:
                count+=1
                print(count)
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
                    lines=f.readlines()[1:]
                    leyeline=lines[0:10]
                    reyeline=lines[10:20]
                    lbrowline=lines[20:28]
                    rbrowline=lines[28:36]
                    noseline=lines[36:46]
                    mouthline=lines[46:64]
                    landmarks=[]


                    #leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[8]]
                    #reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[8]]
                    #lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                    #rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                    #noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]

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




                    #print(len(landmarks))
                    #raw_input('pause')


                    for i in range(64):
                        l=landmarks[2*i]
                        r=landmarks[2*i+1]
                        #print('l:',l)
                        #print('r:',r)
                        #raw_input('pause')
                        #cv2.circle(img,(l,r),2,(0,0,255),-1)
                    #cv2.imshow('img',img)
                    #cv2.waitKey(0)


                    with open(txtfile,'a') as f:
                        f.write(picfile+ ' -1 ')
                        for i in range(64):
                            f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                        f.write('\n')



def get_fplus_clean_build38eye(persondir,cleantxt,txtfile):
    ff=open(cleantxt)
    clines=ff.readlines()
    newclines=[]
    count=0
    #print(clines[:10])
    for cline in clines:
        newcline= cline.strip('\n')
        newcline=os.path.basename(newcline)
        #print(newcline)
        #raw_input('pause')
        newclines.append(newcline)
    print('num of clean data:',len(newclines))
    ff.close()
    for j in os.listdir(persondir):
        #print(j)
        #print(persondir)
        annofile=persondir+'/'+j

        if '_fplus.txt' in annofile:
            #print(annofile)

            picfile=annofile[:-10]+'.png'
            picbsname=os.path.basename(picfile)
            #print(picbsname)
            #raw_input('pause')
            if picbsname in newclines:
                count+=1
                print(count)
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
                    lines=f.readlines()[1:]
                    leyeline=lines[0:10]
                    reyeline=lines[10:20]
                    lbrowline=lines[20:28]
                    rbrowline=lines[28:36]
                    noseline=lines[36:46]
                    mouthline=lines[46:64]
                    landmarks=[]


                    leyeline=[leyeline[0],leyeline[7],leyeline[6],leyeline[5],leyeline[4],leyeline[3],leyeline[2],leyeline[1],leyeline[8]]
                    reyeline=[reyeline[0],reyeline[7],reyeline[6],reyeline[5],reyeline[4],reyeline[3],reyeline[2],reyeline[1],reyeline[8]]
                    lbrowline=[lbrowline[0],lbrowline[7],lbrowline[6],lbrowline[5],lbrowline[4],lbrowline[3],lbrowline[2],lbrowline[1]]
                    rbrowline=[rbrowline[0],rbrowline[7],rbrowline[6],rbrowline[5],rbrowline[4],rbrowline[3],rbrowline[2],rbrowline[1]]
                    noseline=[noseline[0],noseline[1],noseline[8],noseline[7]]

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




                    #print(len(landmarks))
                    #raw_input('pause')


                    for i in range(38):
                        l=landmarks[2*i]
                        r=landmarks[2*i+1]
                        #print('l:',l)
                        #print('r:',r)
                        #raw_input('pause')
                        #cv2.circle(img,(l,r),2,(0,0,255),-1)
                    #cv2.imshow('img',img)
                    #cv2.waitKey(0)


                    with open(txtfile,'a') as f:
                        f.write(picfile+ ' -1 ')
                        for i in range(38):
                            f.write(str(landmarks[2*i])+' '+str(landmarks[2*i+1])+' ')
                        f.write('\n')
if __name__ == '__main__':
    #datadir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/VGG500_hdpose'
    #persondir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/LMNet128result/right'
    #persondir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/close1'
    #persondir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/formalface'
    #persondir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/landmark1203_05clean'
    #persondir='/media/ficha/samsung500/pupildata/rawdata/realenvir'
    #persondir='/media/ficha/samsung500/pupildata/rawdata/eyestatus_add/open_0116_2'
    #persondir='/media/ficha/samsung500/pupildata/rawdata/close_0204/0204_open6-9'
    persondir='/media/ficha/samsung500/pupildata/rawdata/close_0227/eyestatus_0221_2_result/open'


    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/FPeyeclose1.txt'
    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/LM3.txt'
    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/1203_05cleaneye.txt'
    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/LMNet128result/right.txt'
    #txtfile='/media/ficha/samsung500/pupildata/rawdata/realenvir.txt'
    #txtfile='/media/ficha/samsung500/pupildata/rawdata/eyestatus_add/open_0116_2.txt'
    #txtfile='/media/ficha/samsung500/pupildata/rawdata/close_0204/0204_open6-9.txt'
    txtfile='/media/ficha/samsung500/pupildata/rawdata/close_0227/eyestatus_0221_2_result/open_lm11a.txt'

    #get_fplus_build38(persondir,txtfile)
    #get_fplus_build64(persondir,txtfile)
    get_lm11a_build64(persondir,txtfile)


    '''
    #listdir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/pupil-annotation'
    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/LM4.txt'
    #listdir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/inemuri'
    #txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/inemuri.txt'
    listdir='/media/ficha/samsung500/pupildata/rawdata/PKG_1'
    txtfile='/media/ficha/samsung500/pupildata/rawdata/PKG_1.txt'

    for i in os.listdir(listdir):
        persondir=listdir+'/'+i
        print(persondir)
        #get_fplus_build38(persondir,txtfile)
        get_fplus_desay(persondir,txtfile)
    '''

    '''
    listdir='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/landmark1203_05'
    cleantxt='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/sidepngclean.txt'
    txtfile='/media/ficha/2fa9df47-f968-40fc-913f-8204af6ac23e/pupildata/rawdata/sideeyeclean.txt'
    get_fplus_clean_build38eye(listdir,cleantxt,txtfile)
    '''
