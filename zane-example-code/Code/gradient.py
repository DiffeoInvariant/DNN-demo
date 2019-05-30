'''
*******************************************************************************

Author: Zane Jakobs
File purpose: compute requisite derivatives and gradients

*******************************************************************************
'''
import numpy as np
from numba import jit, njit
'''
computes the y-derivative of a grayscale image represented 
as a numpy matrix
'''
@njit(parallel=True)
def imDY(image):
    rows,cols = image.shape()
    dy = np.zeros((rows-2,cols), order='F')
    for c in range(cols):
        for r in range(1,rows-1):
            dy[r,c] = 0.5 * (image[r+1,c] - image[r-1,c])
            
    return dy
    
    
    