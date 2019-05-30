'''
*******************************************************************************

Author: Zane Jakobs
File purpose: finds edges in the image

*******************************************************************************
'''

import numpy as np
import gradient
from numba import jit,njit

#average is 0 if within width/2 of the edge of the row
@njit
def WindowAvg(row, width, rowLength):
    
    avgs = np.zeros(rowLength)
    
    for i in range(rowLength):
        if i >= width/2 and i < rowLength - width/2:
            avgs[i] = np.mean(row[(i-width * 0.5):(i+width * 0.5)])
    
    return avgs
        
        

'''
width is in pixels

orientation options: horiz (horizontal lines)

returns matrix whose first column contains index of row, second contains 
index of middle, third contains value
'''
@njit(parallel=True)
def findMaxDY(image, numEdges,
                width=100):
    dy = gradient.imDY(image)
    rows,cols = dy.shape()
    rowGrads = np.zeros((rows,2))
    #sorted smallest to largest
    maxGrads = np.zeros((numEdges,3))
    #compute largest "bar" in each row by gradient val
    for r in range(rows):
        #window averages centered at each index
        avgs = WindowAvg(dy[r,:], width, cols)
        maxId = np.argmax(np.abs(avgs))
        rowGrads[r,0] = maxId
        maxVal = avgs[maxId]
        rowGrads[r,1] = maxVal
        #check if this is one of the largest gradient values
        for ed in range(numEdges):
            if abs(maxVal) > abs(maxGrads[ed,2]):
                #move all elements less than abs(maxVal) down one
                for v in range(ed):
                    maxGrads[v,0] = maxGrads[v+1,0]
                    maxGrads[v,1] = maxGrads[v+1,1]
                    maxGrads[v,2] = maxGrads[v+1,2]
                maxGrads[ed,0] = r
                maxGrads[ed,1] = maxId
                maxGrads[ed,2] = maxVal
                
    return maxGrads
                