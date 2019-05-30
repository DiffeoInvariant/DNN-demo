import Input, imFilter, gradient,findEdge
import matplotlib.pyplot as plt
import sys
import os
import cv2

'''
*******************************************************************************

Author: Zane Jakobs
File purpose: test code in this repo
*******************************************************************************
'''


def writeVideoFrames():
    import subprocess
    #create directory for frames 
    subprocess.run("mkdir Frames", shell=True)
    #load video, write image files
    Input.MovWriteFrames("../DSC_1690.MOV")
         

def testGetImageSobelFilter(name="Frame3893.jpg", filepath = "Frames/"):
    #check that frames have been written
    filename = filepath + name
    FramesExist = os.path.isfile(filename)
    if not FramesExist:
        print("Error: frames have not been written yet.\n")
        response = input("Would you like to call writeVideoFrames()? (y/n)")
        if response == 'y' or response == 'Y':
            writeVideoFrames()
        elif response == 'n' or response == 'N':
            print("""
            Frames either do not exist or are not in the %s directory.
            Please either call writeVideoFrames or change the arguments to 
            testGetImageSobelFilter to reflect the actual location of the files. 
            (You can do this with the option -fpath=/path/to/frames).
             """, filepath)

    print("Enter ctrl+C when you are done viewing the images.\n")

    '''
    test getImage for rbg OpenCV image
    '''
    frame3893 = Input.getImage(filename)
    frame3893 = frame3893[100:800,700:1300]

    plt.figure(figsize=(16,5))
    plt.subplot(151)
    plt.imshow(frame3893)
    plt.title('Color')

    '''
    test b/w imread, Sobel filter for horizontal and vertical lines
    '''

    f3893BW = Input.getImage(filename, "bw")

    #trim image
    f3893BW = f3893BW[100:800,700:1300]

    plt.subplot(152)
    plt.imshow(f3893BW,cmap=plt.get_cmap('gray'))
    plt.title('Black-and-White (filtered)')
    # bilateral filter
    f3893BW = cv2.bilateralFilter(f3893BW, 9, 75,75)

    f3893Filter = imFilter.SobelHorizontal(f3893BW)

    
    plt.subplot(153)
    plt.imshow(f3893Filter,cmap=plt.get_cmap('gray'))
    plt.title('Horizontal Sobel (post-smoothing')

    f3893FilterVert = imFilter.SobelVertical(f3893BW)

    plt.subplot(154)
    plt.imshow(f3893FilterVert,cmap=plt.get_cmap('gray'))
    plt.title('Vertical Sobel (post-smoothing)')

    #image Laplacian--doesn't give anything useful...
    lapImage = cv2.Laplacian(f3893BW, ddepth=2)
    plt.subplot(155)
    plt.imshow(lapImage, cmap=plt.get_cmap('gray'))
    plt.title("Laplacian filter")
    
    
    plt.show()

def main():
    #check if asking for options
    if len(sys.argv) > 1:
        if sys.argv[1] == '-h' or sys.argv[1] == '-help':
            print("""
                Options: \n, 
                 -h: same as -help, shows this message
                 -handle_frames: creates and writes the movie frames to Frames directory.\n,
                 -image_test: check if frames have been written, tests some image functions.\n 
                 -fpath=/path/to/frames: optional argument in case frames are not in BeerFoam/Frames (must be
                 used with -image_test)\n
                 -fname=[file name]: optional argument of which while to use (default Frame3893.jpg).] (must
                 be used with -image_test)\n
                 ------------------------------------------------------------------------------------------\n

                 Exiting with status 0.\n
                 """)
            sys.exit(0)

        fname = None
        fpath = None
        mode = None
        for args in sys.argv:
            if args.find('-image_test') > -1:
                mode = "image_test"
            if args.find('fpath=') > -1:
                fpath = args.split('fpath=',1)[1]
            if args.find('fname=') > -1:
                fname = args.split('fname=',1)[1]
                     
            if args.find("-handle_frames") > -1:
                 mode = "handle_frames"
        
        if mode == "image_test":
            if not fpath is None:
                if not fname is None:
                    testGetImageSobelFilter(fname, fpath)
                else:
                    testGetImageSobelFilter(filepath=fpath)
            else:
                if not fname is None:
                    testGetImageSobelFilter(fname)
                else:
                    testGetImageSobelFilter()
        elif mode == "handle_frames":
            writeVideoFrames()
        else:
            print("""
            Error: invalid command-line arguments to driver.py. Run with '-h' or '-help' to 
            see list of valid options. Exiting with status 1.
            """)
            sys.exit(1)

    else:
        #if no arguments given
        print("Default: running with -image_test.")
        testGetImageSobelFilter()


if __name__ == "__main__": main()


