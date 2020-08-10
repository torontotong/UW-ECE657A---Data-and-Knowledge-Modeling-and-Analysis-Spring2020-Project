import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,MaxPooling2D
from keras.layers import Dense, Flatten, Reshape, Conv2DTranspose
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from data_preprocessor import *
from pandas.core.frame import DataFrame

img_x = 141
img_y = 110

def MSE(input,output,label, images):
    mse_list = list()
    cos_list = list()
    color = list()
    for l in label:
        if l == 'comp':
            color.append('b')
        if l == 'uncomp':
            color.append('r')
    for k in range(input.shape[0]):
        i = input[k]
        i = i.reshape(141,110)
        o = output[k]
        o = o.reshape(141,110)
        nor_i = normalize(i, norm='l2').flatten()
        nor_o = normalize(o, norm='l2').flatten()
        res = np.array([[nor_i[i] * nor_o[i], nor_i[i] * nor_i[i], nor_o[i] * nor_o[i]] for i in range(len(nor_i))])
        temp = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
        mse = mean_squared_error(i,o)
        mse_list.append(mse)
        cos = 0.5 * temp + 0.5
        cos_list.append(cos)
    df = {'MSE':mse_list,'CosSim':cos_list,'label':label,'color':color, 'image_mname':images}
    data = DataFrame(df)
    colors = {'comp':'b','uncomp':'r'}


    fig1 = plt.figure()
    axes1 = fig1.add_subplot(1, 1, 1)
    scatter1 = axes1.scatter(data['MSE'], data['CosSim'], c=data['color'],s=2)
    axes1.set_title('Two indicator plot of test data')
    axes1.set_xlabel('Loss')
    axes1.set_ylabel('Similarity')
    #axes1.legend(*scatter1.legend_elements(), loc="best", title="label")
    plt.savefig('result.png')
    plt.close(0)
    print(data)
    return mse_list, cos_list


# label: comp/uncomp
def load_data():
    label = list()
    DataMat = list()
    image_list = list()
    dataset_path = './testset/yoda2/comp'
    file_list = listdir(dataset_path)
    m = len(file_list)
    for i in range(m):
        file_name = file_list[i]
        fileStr = file_name.split('.')[0]
        if fileStr == '':
            continue
        inverse_LowPass = fft_Gaussian_LowPass_filer(dataset_path, file_name)
        array = inverse_LowPass.reshape(img_x, img_y)
        if array.min() == 0 and array.max() == 0:
            continue
        DataMat.append(array)
        label.append("comp")
        image_list.append(file_name)
    dataset_path = './testset/yoda2/uncomp'
    file_list = listdir(dataset_path)
    m = len(file_list)
    for i in range(m):
        file_name = file_list[i]
        fileStr = file_name.split('.')[0]
        if fileStr == '':
            continue
        inverse_LowPass = fft_Gaussian_LowPass_filer(dataset_path, file_name)
        array = inverse_LowPass.reshape(img_x, img_y)
        DataMat.append(array)
        label.append("uncomp")
        image_list.append(file_name)
    DataArray = np.array(DataMat)
    DataArray = DataArray.reshape(DataArray.shape[0], img_x, img_y, 1)
    DataArray = DataArray.astype('float32')
    DataArray /= 255
    return DataArray,label, image_list

def draw_decision_buondray(mse_data_lst, cos_data_lst, label, images):
    df = {'MSE':mse_data_lst,'CosSim':cos_data_lst}
    data = DataFrame(df)
    data_array = np.array(data)
    X = data_array
    label_id = []
    for item in label:
        if item == 'comp':
            label_id.append(1)
        elif item == 'uncomp':
            label_id.append(0)

    y = np.array(label_id, np.int)
    h = .0001
    # we create an instance of SVM and fit out data. We do not scale our
    # data since we want to plot the support vectors
    C = 10.0  # SVM regularization parameter
    knn_clf = KNeighborsClassifier(n_neighbors=5).fit(X,y)
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 3*h, X[:, 0].max() + 3*h
    y_min, y_max = X[:, 1].min() - h, X[:, 1].max() + 2*h
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    #for i, clf in enumerate(( rbf_svc, poly_svc)):
    clf = knn_clf
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 1, 1)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=2)
    for i in range(len(X)):
        plt.text(X[i,0], X[i,1], images[i], fontsize='xx-small')

    plt.xlabel('MSE loss')
    plt.ylabel('Cos Similarity')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(np.arange(xx.min(), xx.max(), 5*h))
    plt.yticks(np.arange(yy.min(), yy.max(), 10*h))
    plt.title('KNN=4')
    #plt.show()
    plt.savefig('decision_boundary.png', dpi=300)


def Autoncoder(x_train,label, images):
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                     input_shape=(img_x, img_y, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', strides=(2, 2)))
    #model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu',  strides=(2, 2)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu',  strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))

    # Decoder:
    model.add(Dense(8960, activation='relu'))
    model.add(Reshape((7,5, 256), input_shape=(8960,)))
    model.add(Conv2DTranspose(128, kernel_size=(4, 4), activation='relu', strides=(2, 2)))
    model.add(Conv2DTranspose(64, kernel_size=(4, 4), activation='relu', strides=(2, 2)))
    model.add(Conv2DTranspose(32, kernel_size=(4, 4), activation='relu', strides=(2, 2)))
    model.add(Conv2DTranspose(1,kernel_size=(3, 4), activation='relu', strides=(2, 2)))
    model.compile(optimizer='adam', loss='mse')
    #plot_model(model, to_file='LeNet-5.png', show_shapes=True, show_layer_names=True)
    print(model.summary())
    history = model.fit(x_train, x_train, epochs=100)
    print(1)
    encoded = model.predict(x_train)
    print(2)
    mse_data_lst, cos_data_lst = MSE(x_train,encoded,label, images)
    draw_decision_buondray(mse_data_lst, cos_data_lst, label, images)


def main():
    x_train,label, image_lst= load_data()
    Autoncoder(x_train,label, image_lst)


if __name__ == '__main__':
    main()