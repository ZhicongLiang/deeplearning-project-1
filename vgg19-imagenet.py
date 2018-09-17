'''
reference:
https://github.com/flyyufelix/cnn_finetune
'''
from keras.layers import Dense, Dropout, Flatten
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import cv2
import datetime

def load_reshape_cifar10(num_test):
    '''
    load the cifa10 data and reshape it into 224*224'''
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    img_rows, img_cols  = 224, 224

    if K.image_dim_ordering() == 'th':
        x_test_reshape = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in x_test[num_test:,:,:,:]])
    else:
        x_test_reshape = np.array([cv2.resize(img, (img_rows,img_cols)) for img in x_test[num_test:,:,:,:]])

    # Transform targets to keras compatible format
    y_test_cate = np_utils.to_categorical(y_test[num_test:], num_classes)
    return x_test_reshape, y_test_cate
    
def block5feature(x):
    return base_model.predict(x, batch_size = 50, verbose=True)

def transfer_learning(x_train, y_train):
    '''
    fine-tune the model on cifar-10
    having finished
    '''
    model = base_model
    # adding three fully-connected layers
    model.layers.append(Dropout(0.5))
    model.layers.append(Flatten())
    model.layers.append(Dense(4096, activation='relu'))
    model.layers.append(Dropout(0.5))
    model.layers.append(Dense(4096, activation='relu'))
    model.layers.append(Dropout(0.5))
    model.layers.append(Dense(num_classes, activation='softmax'))
    model.output = model.layers[-1].output
    # training
    batch_size = 16 
    nb_epoch = 10
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_split = 0.2
              )
    return model


if __name__ == '__main__':
    tic = datetime.datetime.now()
    
    # download pre-trained model trained on imagenet
    base_model = VGG19(weights='./models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False)
    
    num_classes = 10
    num_test = 5000
       
    x_test, y_test = load_reshape_cifar10(num_test)
    
    x_feature_block5pool = block5feature(x_test)
        
    toc = datetime.datetime.now()
    print('Feature extractor:', (toc-tic).total_seconds())
    