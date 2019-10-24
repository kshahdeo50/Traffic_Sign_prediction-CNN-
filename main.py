# Importing important librarys for the model training

import numpy as np    # numpy for mathamatical data manupulation
import pickle         # pickle for data loading into the model
import pandas as pd   # loading the data
import matplotlib.pyplot as plt  # For data Visualization
import warnings
import random
warnings.filterwarnings("ignore")  # ignoring the mild warnings

# data Loading from file using Pickle 

with open("./traffic-signs-data/train.p", mode = 'rb') as training_data:
    train = pickle.load(training_data)
with open("./traffic-signs-data/valid.p", mode = 'rb') as validation_data:
    validation = pickle.load(validation_data)
with open("./traffic-signs-data/test.p", mode = 'rb') as test_data:
    test = pickle.load(test_data) 
    
x_train,y_train = train["features"],train["labels"]
x_test,y_test = test["features"],test["labels"]
x_valid,y_valid = validation["features"],validation["labels"]  


# Visualizing the sample Data
i = 100
plt.imshow(x_train[i])
plt.show()

# shuffleing the dataset
from sklearn.utils import shuffle
x_train,y_train = shuffle(x_train,y_train)

plt.imshow(x_train[i])
plt.show()

#normalizing the data and converting it to grayscale

x_train_gray = np.sum(x_train/3,axis = 3,keepdims = True)
x_test_gray = np.sum(x_test/3,axis = 3,keepdims = True)
x_valid_gray = np.sum(x_valid/3,axis = 3,keepdims = True)

x_train_gray_norm = (x_train_gray - 128 )/128
x_test_gray_norm = (x_test_gray - 128 )/128
x_valid_gray_norm = (x_valid_gray - 128 )/128

# visualizing the train data converted into gray scale

i = 600

plt.imshow(x_train_gray[i].squeeze() ,cmap = 'gray')
plt.figure()
plt.imshow(x_train[i])


#model Bulding 
import tensorflow as tf
import tensorflow.keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout


model = Sequential()
model.add(Conv2D(64,3,3 , input_shape = (32,32,1),activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64,activation = 'relu'))
model.add(Dense(43,activation = 'sigmoid'))

model.summary()

model.compile(activation = "adam",
             loss = 'sparse_categorical_crossentropy',
             metrics = ["accuracy"])
            
history = model.fit(x_train_gray_norm,y_train,
                   epochs = 50,
                    verbose = 1,
                    validation_data = (x_valid_gray_norm,y_valid)
                   )  
# validation on test data            
score = model.evaluate(x_test_gray_norm,y_test , verbose = 0)
print("Accuracy is : {:.4f}".format(score[1]))


#visualizing the training of the model
import matplotlib.pyplot as plt
%matplotlib inline
accuracy = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs ,accuracy,'bo', label = "Training Accuracy")
plt.plot(epochs , val_acc ,'b', label = "VAlidation Accuracy")
plt.title("Training And Validation Accuracy")
plt.legend()
plt.figure()
plt.plot(epochs,loss ,'bo',label ="Training Loss" )
plt.plot(epochs, val_loss,'b',label = "Validation Loss")
plt.legend()
plt.show()

#prediction
predicted_classes = model.predict_classes(x_test_gray_norm)
y_true = y_test

#creating confurion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,predicted_classes)
cm

# Visualizing The prediction
for i in range(0,12):
    plt.subplot(4,3,i+1)
    plt.imshow(x_test_gray_norm[i+10].squeeze(),cmap = 'gray',interpolation = 'none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[i+10], y_true[i+10]))
    plt.tight_layout()
            




