from os import listdir

from PIL import Image
from numpy import asarray, ma
import pandas as pd
import numpy as np
import cv2
from math import sqrt,exp
import matplotlib.pyplot as plt

image_file_name = './Dataset/Pixel2_No4_64_uncomp_1.jpeg'

# Split image into R,G,B three color files
def split_image_full_colors_to_sgl_color(org_image_file_name):

    full_color_image = cv2.imread(org_image_file_name,1)
    b,g,r = cv2.split(full_color_image)
    pos = org_image_file_name.find('Pixel')
    new_file_name = org_image_file_name[pos:]
    pos = new_file_name.find('.')
    new_file_name = new_file_name[:pos]
    cv2.imwrite(new_file_name+'_2.jpeg',b)
    print(b.shape)
    cv2.imwrite(new_file_name+'_1.jpeg',g)
    print(g.shape)
    cv2.imwrite(new_file_name+'_0.jpeg',r)
    print(r.shape)
    return r.shape


# Trim the image
def trim_image(file_name):
    image = Image.open(file_name)
    # convert image to numpy array
    data_array = asarray(image)
    print(type(data_array))
    # summarize shape
    panel_mean_half = data_array.mean() / 2
    data_y = data_array.shape[0]
    data_x = data_array.shape[1]
    center_row= int(data_array.shape[0]/2)
    center_col = int(data_array.shape[1]/2)
    for x in range(0,data_x):
        if data_array[center_row, x] > panel_mean_half:
            col_start = x
            print('left = {0}'.format(col_start))
            break
    for x in range(center_col,data_x):
        if data_array[center_row, x] < panel_mean_half:
            col_end = x
            print('right = {0}'.format(col_end))
            break

    for y in range(0,data_y):
        if data_array[y , center_col] > int(panel_mean_half):
            row_start = y
            print('upper = {0}'.format(row_start))
            break

    row_end = data_y - 1
    for y in range(center_row,data_y):
        if data_array[y , center_col].mean() < int(panel_mean_half/2):
            row_end = y
            break
    print('bottom = {0}'.format(row_end))

    trimed_image = image.crop((col_start, row_start+10 ,col_end-1, row_end-10))
    trimed_image.save(file_name)

# Plotting Data histogram
def plot_histogram(fea_np, color):
    df = pd.DataFrame(fea_np)
    fig = df.plot.hist(bins=50).get_figure()
    ax = fig.axes[0]
    ax.set_title('Image data histogram')
    ax.set_xlabel('Data value')
    ax.set_ylabel('Data Value Frequency')
    ax.grid(True, which='both')
    fig.savefig('Image'+'-'+str(color)+'-hist'+'.png')
    return

def data_cleaner(data_array):
    # get Data array size
    clean_dataarray = data_array
    columns = data_array.shape[1]
    rows = data_array.shape[0]
    mean = data_array.mean()
    clean_dataarray[clean_dataarray < (mean/3)] = mean
    return data_array


def load_image_file_to_array(file_name):
    data_array = np.array(Image.open(file_name))
    return data_array


def preprocess_image_file(dataset_path , file_name):
    file_path = dataset_path+file_name
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    fileStr = file_name.split('.')[0]
    output_file_path = 'processed_data/'+ fileStr + '_g.jpeg'
    cv2.imwrite(output_file_path,img)
    file_path = output_file_path
    trim_image(file_path)
    data_array = np.array(Image.open(file_path))
    clean_data = data_cleaner(data_array)
    i = Image.fromarray(clean_data)
    i.save(file_path)

def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

def fft_Gaussian_LowPass_filer(dataset_path, file_name):
    file_path = dataset_path+'/'+file_name
    fileStr = file_name.split('.')[0]
    output_file_path = 'FFTed_data/'+ fileStr + '_fft.png'
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=(141, 110), interpolation=cv2.INTER_CUBIC)
    original = np.fft.fft2(img)
    center = np.fft.fftshift(original)
    LowPassCenter = center * gaussianLP(50, img.shape)
    LowPass = np.fft.ifftshift(LowPassCenter)
    inverse_LowPass = np.fft.ifft2(LowPass)
    #i = Image.fromarray(np.array(inverse_LowPass, np.uint8))
    #i.save(output_file_path)
    #plt.figure(figsize=(1.4, 1.1), constrained_layout=False)
    #plt.subplot(133)
    #plt.imshow(np.abs(inverse_LowPass), "gray")
    #plt.title("Gaussian Low Pass")
    #plt.show()
    return inverse_LowPass

def resize_image(path, filename):
    img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)
    res = cv2.resize(img, dsize=(110,140), interpolation=cv2.INTER_AREA)
#    (h, w) = res.shape[:2]
#    # calculate the center of the image
#    center = (w / 2, h / 2)
#    angle90 = 90
#    scale = 1.0
#    # 270 degrees
#    M = cv2.getRotationMatrix2D(center, angle90, scale)
#    rotated270 = cv2.warpAffine(res, M, (h, w))
    img = res.T
    file_path = path + 'r_' + filename
    cv2.imwrite(file_path, img)

def main():
    dataset_path = 'processed_data/comp/'
    imageFileList = listdir(dataset_path)
    file_num = len(imageFileList)
    for i in range(file_num):
        fileNameStr = imageFileList[i]
        file_path = dataset_path+fileNameStr
        if fileNameStr == '.DS_Store':
            continue
        #resize_image(dataset_path, fileNameStr)
        #preprocess_image_file(dataset_path, fileNameStr)
        fft_Gaussian_LowPass_filer(dataset_path, fileNameStr)

    dataset_path = 'processed_data/uncomp/'
    imageFileList = listdir(dataset_path)
    file_num = len(imageFileList)
    for i in range(file_num):
        fileNameStr = imageFileList[i]
        file_path = dataset_path+fileNameStr
        if fileNameStr == '.DS_Store':
            continue
        #resize_image(dataset_path, fileNameStr)
        #preprocess_image_file(dataset_path, fileNameStr)
        fft_Gaussian_LowPass_filer(dataset_path, fileNameStr)


if __name__ == '__main__':
    main()