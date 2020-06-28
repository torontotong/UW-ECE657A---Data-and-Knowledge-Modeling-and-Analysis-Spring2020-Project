from PIL import Image
from numpy import asarray, ma
import pandas as pd
import numpy as np
import cv2

image_file_name = './Dataset/Pixel2_No4_64_uncomp_1.jpeg'

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


# load the image
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
    columns = data_array.shape[1]
    rows = data_array.shape[0]
    mean = data_array.mean()
    data_array[data_array < (mean/3)] = mean

    # for x in range(0,50):
    #     mean_of_col = cleaned_data[:,x].mean()
    #     for y in range(rows):
    #         if cleaned_data[y,x] < mean_of_col/4:
    #             cleaned_data[y, x] = mean_of_col
    #
    # for x in range(columns-50, columns):
    #     mean_of_col = cleaned_data[:,x].mean()
    #     for y in range(rows):
    #         if cleaned_data[y,x] < mean_of_col/4:
    #             cleaned_data[y, x] = mean_of_col
    return data_array


def load_image_file_to_array(file_name):
    #im = Image.open(file_name)
    #data_array = asarray(im)
    data_array = np.array(Image.open(file_name))
    return data_array

def main():
    file_name_common = 'Pixel2_No4_64_uncomp_1'
    x, y = split_image_full_colors_to_sgl_color(image_file_name)
    for color in range(0,3):
        file_name = file_name_common+'_'+str(color)+'.jpeg'
        trim_image(file_name)
        image_data_array = load_image_file_to_array(file_name)
        clean_data = data_cleaner(image_data_array)
        file_name = file_name_common+'_'+str(color)+'.csv'
        np.savetxt(file_name, clean_data, delimiter=",")


if __name__ == '__main__':
    main()

