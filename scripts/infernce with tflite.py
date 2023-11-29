import tensorflow as tf 
from tensorflow.keras.preprocessing import image
import numpy as np 
interpreter=tf.lite.Interpreter(model_path=r'C:\Users\edrie\Documents\New folder (2)\classfication_mnist_numbers\models\mnist_model.tflite')
interpreter.allocate_tensors()

img_path = r"C:\Users\edrie\Documents\New folder (2)\classfication_mnist_numbers\data_test\download.jpeg"
img=image.load_img(img_path,target_size=(28,28),color_mode='grayscale')
img=image.img_to_array(img)
img=np.expand_dims(img,axis=0)
img /= 255

input_tens_index=interpreter.get_input_details()[0]['index']
interpreter.tensor(input_tens_index)()[0, :, :] = img[0, :, :, 0]

interpreter.invoke()

out_tens=interpreter.get_output_details()[0]['index']
result_pred=interpreter.tensor(out_tens)()[0]

clas_pred=np.argmax(result_pred)
print(clas_pred)