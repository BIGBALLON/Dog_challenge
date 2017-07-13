import keras
import os
import math
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, AveragePooling2D
from keras.initializers import RandomNormal  
from keras import optimizers
from keras.callbacks import LearningRateScheduler
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
from PIL import Image
import numpy as np
import os.path


weight_name = './retrain.h5'
pic_path = './test_pic/'
res_path = './res.txt'
img_rows, img_cols = 112, 112
img_channels = 3
dropout = 0.5
weight_init = 0.0015


def get_he_weight(k,c):
	return math.sqrt(2/(k*k*c))

# -------- build model -------- #
model = Sequential()
# Block 1
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv1', input_shape=(img_rows,img_cols,img_channels)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,64)), name='block1_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

# Block 2
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,128)), name='block2_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

# Block 3
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,256)), name='block3_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

# Block 4
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = get_he_weight(3,512)), name='block4_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

# Block 5
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv1'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv2'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv3'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(get_he_weight(3,512)), name='block5_conv4'))
model.add(BatchNormalization())
model.add(Activation('relu'))

# model modification for cifar-10
model.add(Flatten(name='flatten'))
model.add(Dense(4096, use_bias = True, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='fc_cifa10'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(4096, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='fc2'))  
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(dropout))	  
model.add(Dense(134, kernel_regularizer=keras.regularizers.l2(weight_init), kernel_initializer=RandomNormal(stddev = 0.01), name='predictions_cifa10'))		
model.add(BatchNormalization())
model.add(Activation('softmax'))
model.load_weights(weight_name)

ans_file = open(res_path, "w");

files = os.listdir(pic_path)  
s = []  
for file in files: 
	if not os.path.isdir(file):  
		img = Image.open(pic_path+file)
		ori_w,ori_h = img.size
		new_w = 224.0;
		new_h = 224.0;
		if ori_w > ori_h:
			bs = 224.0 / ori_h;
			new_w = ori_w * bs
			weight = int(new_w)
			height = int(new_h)
			img = img.resize( (weight, height), Image.BILINEAR )
			region = ( weight / 2 - 112, 0, weight / 2 + 112, height)
			img = img.crop( region )
		else:
			bs = 224.0 / ori_w;
			new_h = ori_h * bs
			weight = int(new_w)
			height = int(new_h)
			img = img.resize( (weight, height), Image.BILINEAR )
			region = ( 0, height / 2 - 112 , weight, height / 2 + 112  )
			img = img.crop( region )
		img = img.resize( (112, 112), Image.BILINEAR )
		x = np.array( img, dtype = 'float32' )
		x[:, :, 0] = x[:, :, 0] - 123.680
		x[:, :, 1] = x[:, :, 1] - 116.779
		x[:, :, 2] = x[:, :, 2] - 103.939
		x = np.expand_dims(x, axis=0)
		results = model.predict(x)[0]
		ans = np.argmax(results)
		file,_ = file.split('.')
		ans_file.write(str(ans))
		ans_file.write('\t')
		ans_file.write(file)
		ans_file.write('\n')