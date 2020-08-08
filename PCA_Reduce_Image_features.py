# encoding:GBK
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal

def Cov(dataMat):
    meanVal = np.mean(data, 0)
    meanVal = np.tile(meanVal, (rows, 1))
    Z = dataMat - meanVal
    Zcov = (1 / (rows - 1)) * Z.T * Z
    return Zcov

def Percentage2n(eigVals, percentage):
    sortArray = np.sort(eigVals)
    sortArray = sortArray[-1::-1]
    arraySum = sum(sortArray)
    tmpSum = 0
    num = 0
    for i in sortArray:
        tmpSum += i
        num += 1
        if tmpSum >= arraySum * percentage:
            return num

def EigDV(covMat, p):
    D, V = np.linalg.eig(covMat)
    k = Percentage2n(D, p)
    print("Keep {0}% Information need top {1} components".format(p*100, k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector, k

def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector

def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat

def PCA_function(data, p):
    dataMat = np.float32(np.mat(data))
    dataMat, meanVal = Z_centered(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    D, V , k = EigDV(covMat, p)
    lowDataMat = getlowDataMat(dataMat, V)
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat, k


def main():
    imagePath = './processed_data/comp/r_Pixel2_No4_240_Comp_0_g.jpeg'
    image = cv.imread(imagePath)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    rows, cols = image.shape
    print("降维前的特征个数：" + str(cols) + "\n")
    print(image)
    print('----------------------------------------')
    start_time = time.time()
    reconImage, k = PCA_function(image, 0.99)
    print('PCA run {0} seconds'.format(time.time()-start_time))
    reconImage = reconImage.astype(np.uint8)
    print(reconImage)
    cv.imwrite('./processed_data/uncomp/reconstructed/' + '064_uncomp.png', reconImage)
    plt.figure(constrained_layout=False)
    #plt.subplot(133)
    plt.imshow(np.abs(reconImage), "gray")
    plt.title("Gaussian Low Pass")
    plt.show()


if __name__ == '__main__':
    main()
