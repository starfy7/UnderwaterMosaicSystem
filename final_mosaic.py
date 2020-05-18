# -*- coding: utf-8 -*-

import imutils
import cv2
import numpy as np
from Tkinter import *
import tkFileDialog
import copy
import time
import argparse
import os
from google_vision import google_vision

root = Tk()
root.withdraw()

#parser = argparse.ArgumentParser(description="Perform mosaicking on video.")
#parser.add_argument("directory", nargs='?', default='.',
#                    help="Absolute or relative path to directory containing videos.")
#parser.add_argument("pitch", nargs='?', default=45., type=float, help="Static pitch angle to use.")
#args = parser.parse_args()

# Add Video Directory
imageDir = "C:\Users\zeina\Quaternions"

FilepathFirst = tkFileDialog.askopenfilename(initialdir=imageDir, title="Select file", filetypes=[
    ("all video format", ".mp4"),
    ("all video format", ".MOV"),
    ("all video format", ".avi"),
    ("all video format", ".mkv")
])

firstImagePath = FilepathFirst.encode("ascii", errors='xmlcharrefreplace')


##############  Image Pre-processing #####################
## Color Correction
def fix_color(img, percent=1):
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)


#def fix_contrast(imrgb):
#    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
#
#    imrgb[:, :, 0] = clahe.apply(imrgb[:, :, 0])
#    imrgb[:, :, 1] = clahe.apply(imrgb[:, :, 1])
#    imrgb[:, :, 2] = clahe.apply(imrgb[:, :, 2])
#
#    return imrgb
#
#
#def fix_light(image, limit=3, grid=(7, 7), gray=False):
#    if (len(image.shape) == 2):
#        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#        gray = True
#
#    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
#    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
#    l, a, b = cv2.split(lab)
#
#    cl = clahe.apply(l)
#    limg = cv2.merge((cl, a, b))
#
#    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#    if gray:
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#    return np.uint8(image)


########### Keep Window ration (BO) ###############
def showImageKeepRatio(winName, image, magnification=1.):
    # Shows an image sized with the factor magnification: default is 1
    if cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) == 0.0:
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winName, int(image.shape[1] * magnification), int(image.shape[0] * magnification))
    cv2.imshow(winName, image)

def clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    if img.dtype != np.uint8:
        raise TypeError('clahe supports only uint8 inputs')

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:, :, 0] = clahe.apply(img[:, :, 0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)

    return img 

## Bird View method- get a top down view of the frames
def bird_view(image, pitch=45):
    ## Crop image
    image= fix_color (image)
    IMAGE_H = image.shape[0]
    IMAGE_W = image.shape[1]
    image = image[0:(IMAGE_H-100), 0:IMAGE_W]
    image=clahe(image)


    # Assume that focus length is equal half image width

    FOCUS = IMAGE_W *2  ## Focus needs to stay the same height of the seabed-camera
    warped_img = None
    pRad = pitch * np.pi/180
    sinPt = np.sin(pRad)
    cosPt = np.cos(pRad)
    Yp = IMAGE_H * sinPt
    Zp = IMAGE_H * cosPt + FOCUS
    Xp = -IMAGE_W/2
    XDiff = Xp * FOCUS / Zp + IMAGE_W/2
    YDiff = IMAGE_H - Yp * FOCUS / Zp
        # Vary upper source points
#    src = np.float32([[XDiff, YDiff],[0, IMAGE_H - 1], [IMAGE_H - 1, IMAGE_W - 1],  [IMAGE_W -1,YDiff]]).reshape(-1,1,2)
#    dst = np.float32([[0, 0],[0, IMAGE_H - 1], [IMAGE_H - 1, IMAGE_W - 1],  [IMAGE_W - 1, 0]]).reshape(-1,1,2)
    src = np.float32([[0, IMAGE_H - 1], [IMAGE_W - 1, IMAGE_H - 1], [XDiff, YDiff], [IMAGE_W - 1-XDiff , YDiff]]).reshape(-1,1,2)
    dst = np.float32([[0, IMAGE_H - 1], [IMAGE_W - 1, IMAGE_H - 1], [0, 0], [IMAGE_W - 1, 0]]).reshape(-1,1,2)

    transformation,_ = cv2.findHomography(src, dst,cv2.RANSAC )
    warpedCorners = cv2.perspectiveTransform(src, transformation) # The transformation matrix
    list_of_points = np.concatenate(( dst, warpedCorners), axis=0)
    

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min,-y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]])
    warped_img = cv2.warpPerspective(image, H_translation.dot(transformation), (x_max-x_min, y_max-y_min),flags=cv2.INTER_NEAREST)
#    FrameSize = output_img.shape
#    NewImage = image.shape
    warped_img= warped_img[translation_dist[1]:IMAGE_H+translation_dist[1], translation_dist[0]:IMAGE_W+translation_dist[0]] 
 
    return  warped_img

def alignImages(im1, im2, detector):

    
    ## Turn frames to grayscale
    gray1 = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
    
    ##  Get the binary masks from the grayscaled frames
    ret1, mask1 = cv2.threshold(gray1,1,255,cv2.THRESH_BINARY)
    ret2, mask2 = cv2.threshold(gray2,1,255,cv2.THRESH_BINARY)

    ## Find features between frames using binary masks too
    kp1, descriptors1 = detector.detectAndCompute(gray1,mask1)
    kp2, descriptors2 = detector.detectAndCompute(gray2,mask2)

    if len(kp1) == 0 or len(kp2) == 0:
        return False, None
    ## Use Brute Force Matcher
    matcher = cv2.BFMatcher()
    
    ## Nearest Neighbor Distance Ratio: The nearest neighbor distance ratio (NNDR),
    ## or ratio test, finds the nearest neighbor to the feature descriptor and second nearest neighbor
    ## to the feature descriptor and divides the two.
    matches = matcher.knnMatch(descriptors2,descriptors1, k=2)  
    
    good = []
    for m, n in matches:
        if m.distance < 0.72 * n.distance: ## Lowe's Ratio ## The ratio inputed here monitors the number of matches found
            ## The smaller the number the less the matchers the stricter the ratio test
            good.append(m)

    print (str(len(good)) + " Matches were Found")

### Detect that at least 10 or less features have been found or else the stitching cannot continue
#    if len(good) <= 4:
#        return im1

    ## Copy the good matches in matches
    if len(good) <= 10:
        return good
    
    ## Define source and destination points between the images based on the good matches
    src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    ## Find the affine transformation between the source and destination points 
    A,mask = cv2.estimateAffine2D(src_pts,dst_pts)

    ## If the affine transofrmation is none -> Find homography matrix instead
    
    H,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC) ## dst first or src??
    if H is None:
                print('No homography Found')
  
    
    ## Find heigh and width of both frames
    h1,w1 = im1.shape[:2]
    h2,w2 = im2.shape[:2]
    ## Fine the corners of frame 1 and frame 2
    corners1 = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
    corners2 = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    ## Warped the corners in an empty black frame
    warpedCorners2 = cv2.perspectiveTransform (corners2, H)   

    corners = np.concatenate((corners1,warpedCorners2),axis=0)
    
    ## Fine the min and max offset
    [xmin, ymin] = np.int32(corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(corners.max(axis=0).ravel() + 0.5)
#    transform_dist = [-xmin, -ymin]

    ## Calculate translation
    translation = np.float32(([1,0,-1*xmin],[0,1,-1*ymin],[0,0,1]))
    
    ## Warpe the Res frame based on the first frame, the trasnaltion and difference between offset
    warpedResImg = cv2.warpPerspective(im1, translation, (xmax-xmin, ymax-ymin),flags=cv2.INTER_NEAREST+ cv2.WARP_FILL_OUTLIERS,borderMode=cv2.BORDER_TRANSPARENT)
        ## Again if Affine transformation is none
        ## Warping will be with the dot product of the translation and the homography matrix
    fullTransformation = np.dot(translation,H) #again, images must be translated to be 100% visible in new canvas
    warpedImage2 = cv2.warpPerspective(im2, fullTransformation, (xmax-xmin, ymax-ymin),flags=cv2.INTER_NEAREST+ cv2.WARP_FILL_OUTLIERS,borderMode=cv2.BORDER_TRANSPARENT, borderValue=(0, 0, 0))
#
    
    result = np.where(warpedImage2 != 0, warpedImage2, warpedResImg)

##        # transform the panorama image to grayscale and threshold it 
#    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
#    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
#    
#    # Finds contours from the binary image
#    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#    cnts = imutils.grab_contours(cnts)
#    
#    # get the maximum contour area
#    c = max(cnts, key=cv2.contourArea)
#    
#    # get a bbox from the contour area
#    (x, y, w, h) = cv2.boundingRect(c)
#    
#    # crop the image to the bbox coordinates
#    result = result[y:y + h, x:x + w]
    return True, result



def createMosaic(video, pitch=45):
    if len(video) == 0:
        print "No video chosen!"
        return

    # Start time
    start = time.time()
    keep_processing = True
    #    count = 0
    detector = cv2.xfeatures2d.SIFT_create(3000)

    # define video capture object
    cap = cv2.VideoCapture(video)
    video_out = os.path.join(imageDir, "Seabed-Fish.avi")
    writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps=5.0, frameSize=(640, 240), isColor=True)
    frame_number = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Total Frame Number: {:d}'.format(int(frame_number)))

    # initially mosaic is None
    mosaic = None

    ## Read first frame
    ret, in_frame = cap.read()
    first_frame = cv2.resize(in_frame, (0, 0), fx=0.5, fy=0.5)
    input_frame = cv2.resize(in_frame, (320, 240))

    #    c1=fix_contrast(first_frame)
#    input_frame_col = fix_color(input_frame)
#    l1 = fix_light(first_frame_col)
    b1 = bird_view(in_frame, pitch)
    mosaic = b1

    ## Set mosaic to first frame


    windowNameLive = "Video Input"  # window name
    windowNameMosaic = "Mosaic Output"
    windowNameCombined = "Combined View"

    # create windows by name (as resizable)
    cv2.namedWindow(windowNameLive, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameMosaic, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowNameCombined, cv2.WINDOW_NORMAL)
    stall = 0
    while (keep_processing):
        if (cap.isOpened):
            ## Read all the other frames
            ret, frame = cap.read()
            showImageKeepRatio(windowNameLive, frame, 0.5)

            ## Frame Processsing
            # for the video
            input_frame = cv2.resize(frame, (320, 240))
#            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#            frame_col = fix_color(frame)
#            l2 = fix_light(frame_col)
            b2 = bird_view(frame, pitch)

            ## Image Registration and Stitching
            success, result = alignImages(mosaic, b2, detector)
            if success:
                mosaic = result
                stall = 0
            else:
                stall = stall + 1
            input_mosaic = cv2.resize(mosaic,(320, 240))
            showImageKeepRatio(windowNameMosaic, mosaic, 0.5)

            ## Combined image to video
            video_frame = cv2.hconcat([input_frame, input_mosaic])
            writer.write(video_frame)
            showImageKeepRatio(windowNameCombined, video_frame)

        # continue to next frame (i.e. next loop iteration)

        # when we reach the end of the video (file) exit cleanly

        if not ret:
            keep_processing = False
            continue
        if stall > 10:
            keep_processing = False
            continue

        key = cv2.waitKey(10) & 0xFF
        if (key == ord('x')):
            keep_processing = False
            cv2.destroyWindow(windowNameMosaic)
            cv2.destroyWindow(windowNameLive)
            cv2.destroyWindow(windowNameCombined)

    writer.release()
    ## Save mosaic
    cv2.imwrite('Final Mosaic.jpg', result)
    ## Google Vision Part ##
#    image = 'Picture1.jpg'
#    google_vision(image)
    end = time.time()
    # Time elapsed
    seconds = end - start
    print "Time taken : {0} seconds".format(seconds)

    cap.release()


root.destroy()

#createMosaic(FilepathFirst)
createMosaic('ROV3.mp4')