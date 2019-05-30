'''
*******************************************************************************

Author: Zane Jakobs
File purpose: filter images

*******************************************************************************
'''

from scipy import ndimage

def SobelHorizontal(npImage):
    return ndimage.sobel(npImage, axis=0, mode='constant')

def SobelVertical(npImage):
    return ndimage.sobel(npImage, axis=1, mode='constant')


