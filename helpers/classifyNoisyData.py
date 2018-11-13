import time
import hashlib
import scipy
import argparse
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from gap_statistic import OptimalK
import os
from shutil import copyfile

ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="Path to the input image")
# ap.add_argument("-m", "--method", required=True, help="Sorting method")
args = vars(ap.parse_args())


def readImg(imPath):
    import matplotlib.image as mpimg
    import cv2
    img = cv2.imread(imPath)
    img = cv2.GaussianBlur(img,(11,11),7)
    img = cv2.pyrMeanShiftFiltering(img, 30, 51)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def filterImg(img):
    # reshape the image
    reshapedImg = np.reshape(img, (img.shape[0]*img.shape[1], 3))
    # filtering: remove black colors
    finalImg = reshapedImg[np.any(reshapedImg, axis=1)]
    # flatten all pixels values under 25
    thresh = 25
    finalImg[(finalImg < thresh).all(axis=1)] = 0
    
    removed = (reshapedImg.shape[0] - finalImg.shape[0]) * 100 / reshapedImg.shape[0]
    return finalImg


def zScore(finalImg):
    
    from scipy import stats
    import numpy as np
    z = np.abs(stats.zscore(finalImg))
    return z


def removeOutliers(finalImg, z, threshold):
    threshold = 1.8
    finalImg = finalImg[(z < threshold).all(axis=1)]
    return finalImg

def findOptimalK(finalImg):
    optimalK = OptimalK(parallel_backend='rust')
    n_clusters = optimalK(finalImg, cluster_array=np.arange(1,4))
    print('Optimal clusters: %d'% n_clusters)
    return n_clusters


folderPath = r"C:\Users\sohil\AnacondaProjects\JobTask\example_output\\"
threshold = 1.8

# Create directory
dirName = 'NoisyData'
try:
    # Create target Directory
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ") 
except FileExistsError:
    print("Directory " , dirName ,  " already exists")
    
for i in range(1,1965):
    i = 1
    imgPath = folderPath + "extract_field%d.jpg" % i
    img = readImg(imgPath)
    filtered = filterImg(img)
    zscore = zScore(filtered)
    filtered = removeOutliers(filtered, zscore, threshold)
    K = findOptimalK(filtered)
    if K > 2:
        print("Noisy Data")
        src = imgPath
        dst = dirName + "\\%i.png"%i
        copyfile(src, dst)