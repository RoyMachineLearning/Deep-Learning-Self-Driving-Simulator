from __future__ import division
import numpy as np
import pickle
import cv2
import scipy
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy.misc
import math

features_directory = 'Data/'
labels_file = 'labels.txt'


def image_preprocessing(img):
    resized_image = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return resized_image

def Load_Train():
    X = []
    y = []
    features = []

    with open(labels_file) as fp:
        for line in fp:
            X.append(features_directory + line.split()[0])
            y.append(float(line.split()[1]) * scipy.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(image_preprocessing(img))
    
    return features, y


features, labels = Load_Train()

features = np.array(features).astype('float32')
labels = np.array(labels).astype('float32')

with open("features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)



num_images = len(features)-1
print(num_images)
#import model

model = load_model("1. CNN Baseline/SelfDriving.h5")

def keras_predictor(model,image):
    processed = keras_image_processor(image)
    steer_angle = float(model.predict(processed, batch_size=1))
    steer_angle = steer_angle *180/scipy.pi
    return steer_angle

def keras_image_processor(img):
    image_x = 100
    image_y = 100
    img = cv2.resize(img,(image_x,image_y))
    img = np.array(img,dtype=np.float32)
    img = np.reshape (img, (-1,image_x,image_y,1))
    return img


i = math.ceil(num_images*0.8)
steer = cv2.imread("drive_wheel.jpg",0)
rows,cols = steer.shape
smoothed_angle = 0

while (cv2.waitKey(10) != ord('q')):
    #test images as video
    full_image = scipy.misc.imread(features_directory + str(i) +".jpg",mode = "RGB")
    #image = scipy.misc.imresize(full_image[:],[40,40])
    cv2.imshow("FRAME", cv2.cvtColor(full_image,cv2.COLOR_RGB2BGR))
    
    #Now predict the angle for steering wheel
    gray = cv2.resize((cv2.cvtColor(full_image,cv2.COLOR_RGB2HSV))[:,:,1],(100,100))
    
    steering_angle = keras_predictor(model,gray)

    print("steering angle:"+str(steering_angle)+"(Pred) "+ str(labels[i]*180/scipy.pi)+"(actuals)")
    
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(steering_angle - smoothed_angle)
    
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    
    if (i==15100):
        break;

    i += 1
    
 
cv2.destroyAllWindows()


