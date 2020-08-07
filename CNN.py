import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D,MaxPooling2D
from keras.layers import Dense, Flatten, Reshape, Conv2DTranspose
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from sklearn.model_selection import train_test_split


def data_reshape(DataMat): #DataMat: list[array([...],dtype=uint8)]
    stack = list()
    for data in DataMat:
        array = data.reshape(140,110)
        #im = Image.fromarray(array)
        #im.save("your_file.jpeg")
        stack.append(array)
    DataArray = np.array(stack)
    DataArray = DataArray.reshape(DataArray.shape[0], 140, 110, 1)
    DataArray = DataArray.astype('float32')
    DataArray /= 255
    return DataArray

def data_split(image, label):
    x_train,x_test = train_test_split(image, test_size=0.3, random_state=42, shuffle=True)
    y_train, y_test = train_test_split(label, test_size=0.3, random_state=42, shuffle=True)
    return x_train,y_train,x_test,y_test

def Autoncoder():
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(2, 2),
                     input_shape=(140, 110, 1)))
    #model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #plot_model(model, to_file='LeNet-5.png', show_shapes=True, show_layer_names=True)
    print(model.summary())


def main():
    Autoncoder()


if __name__ == '__main__':
    main()