import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.join('input/garbage-classification/TRAIN')
labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic']

for label in labels:
    directory = os.path.join(train_dir, label)
    print("Images of label \"" + label + "\":\t", len(os.listdir(directory)))

plt.figure(figsize=(30,14))

for i in range(6):
    directory = os.path.join(train_dir, labels[i])
    for j in range(10):
        path = os.path.join(directory, os.listdir(directory)[j])
        img = mpimg.imread(path)
        
        plt.subplot(6, 10, i*10 + j + 1)
        plt.imshow(img)
        
        if j == 0:
            plt.ylabel(labels[i], fontsize=20)
        
plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
plt.tight_layout()
plt.show()

directory = os.path.join(train_dir, 'cardboard')
path = os.path.join(directory, os.listdir(directory)[0])
image = mpimg.imread(path)
image.shape

model = tf.keras.models.load_model("../input/lastone/lastone.h5")

model.summary()

train_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True,
                                   rotation_range=15,zoom_range=0.1,
                                   width_shift_range=0.15,height_shift_range=0.15,
                                   shear_range=0.1,
                                   fill_mode="nearest",
                                   rescale=1./255., 
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 200), batch_size=32, class_mode = 'binary', subset='training')
validation_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 200), batch_size=32, shuffle = True, class_mode = 'binary', subset='validation')


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

callbacks = myCallback()

history = model.fit(train_generator, epochs=30, verbose=1, validation_data=validation_generator, callbacks=[callbacks])

model.save("lastone.h5")



feature_dim = 2 # dimension of each data point
training_dataset_size = 20
testing_dataset_size = 10

sample_Total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=training_dataset_size, 
    test_size=testing_dataset_size, 
    n=feature_dim, gap=0.3, plot_data=True
)
datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
print(class_to_label)

aqua_globals.seed = 30
result = SklearnSVM(training_input, test_input, datapoints[0]).run()
print("kernel matrix during the training:")
kernel_matrix = result['kernel_matrix_training']
img = plt.imshow(np.asmatrix(kernel_matrix), interpolation='nearest', origin='upper', cmap='bone_r')
plt.show()

print("testing success ratio: ", result['testing_accuracy'])
print("predicted classes:", result['predicted_classes'])

sample_Total, training_input, test_input, class_labels = breast_cancer(
    training_size=20, test_size=10, n=2, plot_data=True
)
# n =2 is the dimension of each data point

datapoints, class_to_label = split_dataset_to_data_and_labels(test_input)
label_to_class = {label:class_name for class_name, label in class_to_label.items()}
print(class_to_label, label_to_class)















































from PIL import Image
#cat = int(input('Enter any category by index: '))
#ind = int(input('Enter any index to test: '))
for cat in range(6):
    directory = os.path.join(train_dir, labels[cat % 6])
    for ind in range(10, 15):
        path = os.path.join(directory, os.listdir(directory)[ind])
        img = Image.open(path)
        img = img.resize((150, 200))
        display(img)
        x = keras.preprocessing.image.img_to_array(img)
        x = x/255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images)
        pred = labels[np.argmax(classes)]
    
        print("Actual: ", labels[cat % 6], " Prediction: ", pred)


import pyautogui
import PySimpleGUI as sg
import cv2
import numpy as np


def main():

    sg.theme('Black')

    # define the window layout
    layout = [[sg.Text('OpenCV Demo', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Image(filename='', key='image')],
              [sg.Button('Record', size=(10, 1), font='Arial 14'),
               sg.Button('Stop', size=(10, 1), font='Arial 14'),
               sg.Button('Exit', size=(10, 1), font='Arial 14'),
               sg.Button('Screenshot',size=(10,1),font='Arial 14') ]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration',
                       layout, location=(800, 400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv2.VideoCapture(0)
    recording = False

    import serial                                                              #Serial imported for Serial communication
    import time                                                                #Required to use delay functions   
    ArduinoUnoSerial = serial.Serial('com4',9600)       #Create Serial port object called ArduinoUnoSerialData time.sleep(2)                                                             #wait for 2 secounds for the communication to get established
    print(ArduinoUnoSerial.readline())                             #read the serial data and print it as line 
    print ("You have new message from Arduino")
    on = "1"
    arr_on = bytes(on, 'utf-8')
    off = "0"
    arr_off = bytes(off, 'utf-8')

    arr_pred = bytes(pred, 'utf-8')

    while True:
        event, values = window.read(timeout=20)
        if event == 'Exit' or event == sg.WIN_CLOSED:
            return

        elif event == 'Record':
            recording = True
        elif event=='Screenshot':
            myScreenshot = pyautogui.screenshot()
            myScreenshot.save(r'shot.png')

        elif event == 'Stop':
            recording = False
            img = np.full((480, 640), 255)
            # this is faster, shorter and needs less includes
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window['image'].update(data=imgbytes)

        if recording:
            ret, frame = cap.read()
            imgbytes = cv2.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)
        
        ArduinoUnoSerial.write(arr_pred) 

main() 



while 1:
    ArduinoUnoSerial.write(arr_pred)     