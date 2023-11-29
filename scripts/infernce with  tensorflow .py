import tensorflow as tf 
from tensorflow.keras.preprocessing import image 
import matplotlib.pyplot as plt 
import numpy as np 
model = tf.keras.models.load_model(r'C:\Users\edrie\Documents\New folder (2)\classfication_mnist_numbers\models\model.h5')
print(model.summary())
img_path = r"C:\Users\edrie\Documents\New folder (2)\classfication_mnist_numbers\data_test\download.jpeg"
img=image.load_img(img_path,target_size=(28,28),color_mode='grayscale')
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
img /= 255
pred=model.predict(img)
clas_pred=np.argmax(pred)
print(clas_pred)
