import csv
import cv2
import numpy as np 
#grab information for images and driving from .csv generated during training 
data = []
with open("/opt/driving_log.csv") as csvfile: 
    reader = csv.reader(csvfile)
    for line in reader:
        data.append(line)
#create lists to store the images and the measurements  
images = []
measurements = []
#iterate through the data and store the images for center, left, and right camera plus steering
for line in data:
    image_path = line[0]
    filename = image_path.split('/')[-1]
    current_path = '/opt/IMG/'+filename
    image = cv2.imread(current_path)
    
    img_left = cv2.imread('/opt/IMG/'+line[1].split('/')[-1])
    img_right = cv2.imread('/opt/IMG/'+line[2].split('/')[-1])
    
    #crop image so road becomes on the main focus 
    crop_image = image[60:135,:,:]
    crop_left = img_left[60:135,:,:]
    crop_right = img_right[60:135,:,:]
    
    #resize image to match size in nVidia paper 
    resize_image =  cv2.resize(crop_image,(200,66))
    resize_left = cv2.resize(crop_left,(200,66))
    resize_right = cv2.resize(crop_right,(200,66))
    
    #convert from BGR to YUV as outlined in nVidia reserach paper
    image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2YUV)
    img_left = cv2.cvtColor(resize_left, cv2.COLOR_BGR2YUV)
    img_right = cv2.cvtColor(resize_right, cv2.COLOR_BGR2YUV)
    
    #store images 
    #images.append(image)
    images.extend([image,img_left,img_right])
    measurement = float(line[3])
    
    #add side camera steering correction 
    correction = 0.3 # this is a parameter to tune
    steering_left = measurement + correction
    steering_right = measurement - correction
    measurements.extend([measurement, steering_left,steering_right])
    #measurements.append(measurement) 
#augment the data by flipping them and the steering direction. Double data size 
aug_img,aug_msure = [],[]
for image,measurement in zip(images,measurements):
    aug_img.append(image)
    aug_msure.append(measurement)
    aug_img.append(cv2.flip(image,1))
    aug_msure.append(measurement*-1.0)
#create training data arrays     
X_train = np.array(aug_img)
y_train = np.array(aug_msure) 
    
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten,Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
#Model is based on nVidia research paper on End to End Deep Learning 
model = Sequential()
model.add(Lambda(lambda x: x / 255,input_shape=(66,200,3))) #Normalization layer
#model.add(Cropping2D(cropping = ((70,25),(0,0))))

model.add(Conv2D(24,5,5,subsample=(2,2), activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2), activation='relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(48,5,5,subsample=(2,2), activation ='relu'))
#model.add(MaxPooling2D()) #Maxpooling did not work well in the Nvidia architecture setup so was commented out 
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Conv2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(25))
#model.add(Dropout(0.5)) #Overfitting reduction through gathering a lot of training images
#augmenting the data and running for only 4 epochs  was enough and dropout did not result in
#an improvement so was commented out
model.add(Dense(1))

model.compile(loss = 'mse',optimizer = 'adam')
model.fit(X_train, y_train, epochs=4, validation_split=0.2,shuffle = True)

model.save('model1.h5')
exit()



