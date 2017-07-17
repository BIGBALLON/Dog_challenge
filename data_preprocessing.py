#-*-coding:utf-8-*- 
# __author__ = BIGBALLON
# __copyright__ = Copyright 2017-2018, MIT

from PIL import Image
import numpy as np
from numpy import *
import time
import pickle

train_pic_path = "./train_ima_no_val/"
train_pic_val_path = "./train_ima_val/"

def get112image(img_path):
    img = Image.open(img_path)
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
#     x = np.array( img, dtype = 'float32')
    x = np.array(img)
#     test = Image.fromarray(uint8(x), 'RGB')
#     test.show()
#     img.show()
    return x


train_list = []
y_list = []
cnt = 0
start = time.time()
with open("./shuf_train.txt", "r") as f:
    for line in f:
        name,class_id,_ = line.split(' ')
        picname = train_pic_path + name + ".jpg"
        x = get112image(picname)
        train_list.append(x)
        y_list.append(class_id)
        cnt = cnt + 1
        if cnt % 500 == 0:
            print("images: %d, name = %s, class_id = %s" %(cnt,name,class_id))
        if cnt == 2000:
            break;
        
print("images: %d" %(cnt))
elapsed = time.time() - start
print("Time taken: %.3f seconds." %(elapsed))

############################################
# just for testing
############################################

# y = np.array(train_list,dtype = 'float32')
# print(y.shape)
# y[0,:,:,:]

# with open("traindata", "wb") as fp:   #Pickling
#     pickle.dump(train_list, fp)
    
# with open("traindata", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)
# print(np.array(b).shape)


cnt = 0
start = time.time()
with open("./shuf_train_val.txt", "r") as f:
    for line in f:
        name,class_id,_ = line.split(' ')
        picname = train_pic_val_path + name + ".jpg"
        x = get112image(picname)
        train_list.append(x)
        y_list.append(class_id)
        cnt = cnt + 1
        if cnt % 500 == 0:
            print("images: %d, name = %s, class_id = %s" %(cnt,name,class_id))
        if cnt == 2000:
            break;
print("images: %d" %(cnt))
elapsed = time.time() - start
print("Time taken: %.3f seconds." %(elapsed))

train_list = np.array(train_list,dtype = 'float32')
y_list = np.array(y_list,dtype = 'float32')
with open("train_x", "wb") as fp:   #Pickling
    pickle.dump(train_list, fp)

with open("train_y", "wb") as fp:   #Pickling
    pickle.dump(y_list, fp)
