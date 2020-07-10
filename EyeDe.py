
import cv2
import numpy as np
import os
import glob
import copy

#Canny edge function
def imgCanny(img):
    canny = cv2.Canny(img,100,200)
    #here is to show the result of the canny edge just remove the '#'
    #cv2.imshow('Canny',canny)
    return canny

#Sobel edge function
def imgSol(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    sobel=sobelx+sobely
    #here is to show the result of the sobel edge just remove the '#'
    #cv2.imshow('Sobel',sobel)
    return sobel
#image dilate function
def imgDil(img):
    #i tried many values of kernel and this was good for all photos
    kernel = np.ones((3,1), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    #here is to show the result of the dilational function just remove the '#'
    #cv2.imshow('Dilation',dilation)
    return dilation
#image erosion fucntion
def imgEros(img):
    kernel = np.ones((3,1), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    #here is to show the result of the dilational function just remove the '#'
    #cv2.imshow('Erosion',erosion)
    return erosion

#Hough Lines function

#this function is to read the whole images at once instead of read one image by one like i did in the 1st Assinment
#you can change this dir with your own dir
images = []
for img in glob.glob("C:\\Users\\Anas\\Desktop\\2\\Dataset/*.bmp"):
    n= cv2.imread(img,0)
    images.append(n)
#read the images from the list
for i in range(0, len(images)):
    img2 = images[i]
    img3 = copy.deepcopy(img2)
    cannyedge=imgCanny(img2)
    sol=imgSol(img2)
    dil=imgDil(cannyedge)
    eros=imgEros(cannyedge)
#here is houghlines i know its not good but this was my way to insure that all of images are working
    lines = cv2.HoughLinesP(dil,rho = 1,theta = 1*np.pi/180,threshold = 90,minLineLength = 90,maxLineGap = 30)
    for line in lines:
       for x1,y1,x2,y2 in line:
           cv2.line( img3,( x1,y1 ),( x2,y2 ),( 0,255,0 ),2 )
#here is the curved houghlines i took this algorthm from www.learnopencv.com i changed the values to work with my images and implement it here
    circles = cv2.HoughCircles(cannyedge, cv2.HOUGH_GRADIENT, 1, cannyedge.shape[0] / 3, param1=200, param2=10,
                               minRadius=300, maxRadius=320)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for j in circles[0, :]:
            cv2.circle(img2, (j[0], j[1]), j[2], (0, 255, 0), 3)
    #displying lines
    cv2.imshow('Lines',img3)
    #displying curves
    cv2.imshow('Curve',img2)
#each image will be displayed for 3 sec
    cv2.waitKey(3000)
cv2.destroyAllWindows()
cv2.waitKey(0)
cv2.destroyAllWindows()
