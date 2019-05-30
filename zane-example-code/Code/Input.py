'''
*******************************************************************************

Author: Zane Jakobs
File purpose: read in video, split up into images, chunk based on threads
if needed

*******************************************************************************
'''
import cv2 # OpenCV
import imageio # reading into numpy
#from skimage import color
import numpy as np
from numba import jit

'''
Supported imCol parameters: rgb, bw (grayscale)
Supported imType parameters: OpenCV, numpy
'''
def getImage(filename, imCol="rgb", imType="OpenCV"):
    if imType == "OpenCV":
        if imCol == "rgb":
           img = cv2.imread(filename)
           return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif imCol == "bw":
            img = cv2.imread(filename)
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            print("Error: %s is an unsupported imCol parameter.",imCol)
    elif imType == "numpy":
        if imCol == "rgb":
            return np.asfortranarray(imageio.imread(filename))
        elif imCol == "bw":
            img = imageio.imread(filename)
            raise NotImplementedError
            #return np.asfortranarray(color.rgb2gray(img))
        else:
            print("Error: %s is an unsupported imCol parameter.",imCol)
    else:
            print("Error: %s is an unsupported imType parameter.",imType)
            
@jit
def getFrameBlock(start, end, color="rgb"):
    assert start >= end and start >= 0
    imgList = []
    for i in range(start, end+1):
        filename = "File" + str(i) + ".jpg"
        imgList.append(getImage(filename, color))
    return imgList
            
            
def MovWriteFrames(movFilename):
    vid = cv2.VideoCapture(movFilename)
    success,image = vid.read()
    frame = 0
    success = True
    while success:
       # print("writing")
        cv2.imwrite("Frames/Frame%d.jpg" % frame, image)
        success, image = vid.read()
        frame += 1

    

