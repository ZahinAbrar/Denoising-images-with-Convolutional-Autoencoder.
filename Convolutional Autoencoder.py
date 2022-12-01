from keras.datasets import mnist
import tensorflow as tf
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np
from keras import optimizers
import os
import matplotlib.pyplot as plt
from PIL import Image
import glob
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python.platform import gfile
path = 'D:\Fall_2018\Research\Simulation\Limited_Dataset'
images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.jpg']

image_list = []
for image in images:
    image_list.append(image)


resized_image_array=[]
for image_loc in image_list:
    image_contents = tf.read_file(image_loc)
    image_decoded = tf.image.decode_jpeg(image_contents,channels=1)  ## Decode a JPEG-encoded image to a uint8 tensor.
    resized_image = tf.reshape(tf.image.resize_images(image_decoded, [64,64]),[64,64,1])   # single image autoencoder run I need to change it from (28,28,1) to (1,28,28,1)
    resized_image_array.append(resized_image)

### creating Training Dataset
X_train_orig = []
for i in range(200):
  q = resized_image_array[i]
  # print(q)    #Tensor("Log:0", shape=(), dtype=float32)  # printing this is not important
  X_train_orig.append(q)
  
##creating_Training_Image  
X_train_orig = tf.stack(X_train_orig)
print("Stacked X_train_orig shape is", X_train_orig.shape)


corrupted_resized_image_array=[]
noise_factor = 0.000000001
for image_loc in image_list:
    image_contents = tf.read_file(image_loc)
    image_decoded = tf.image.decode_jpeg(image_contents,channels=1)   ## Decode a JPEG-encoded image to a uint8 tensor.
    resized_image = tf.reshape(tf.image.resize_images(image_decoded, [64,64]),[64,64,1])             # single image autoencoder run I need to change it from (28,28,1) to (1,28,28,1)
    corrupted_resized_image = resized_image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=resized_image.shape)
    corrupted_resized_image_array.append(corrupted_resized_image)
    

X_train_noisy = []
for i in range(200):
  q = corrupted_resized_image_array[i]
  # print(q)    #Tensor("Log:0", shape=(), dtype=float32) # printing this is not important
  X_train_noisy.append(q)
X_train_noisy = tf.stack(X_train_noisy)
print("Stacked X_train_noisy shape is", X_train_noisy.shape)
print("Shape of sample_size", X_train_noisy.shape[0] )
batch_size = 20;
num_batches = X_train_noisy.shape[0] // batch_size




X_Input = tf.placeholder(tf.float32, shape = (None,64,64,1))
X_Input_Noise = tf.placeholder(tf.float32, shape = (None, 64,64,1), name = "Input_placeholder")

X_Input_test = tf.placeholder(tf.float32, shape = (None, 64,64,1))
X_Input_test_Noise = tf.placeholder(tf.float32, shape = (None, 64,64,1))

print("Encoder Arcitecture")
x1 = tf.layers.conv2d(inputs= X_Input_Noise, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
print("Enc_layer1", x1.shape)
x2 = tf.layers.max_pooling2d(x1, pool_size = [2,2], padding="same", strides = 2)
print("Enc_layer2", x2.shape)
x3 = tf.layers.conv2d(inputs= x2, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 
print("Enc_layer3", x3.shape)
x3_= tf.layers.max_pooling2d(x3, pool_size = [4,4], padding="same", strides = 2)
print("Enc_layer4", x3_.shape) 
x4 = tf.layers.conv2d(inputs= x3_, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 
print("Enc_layer5", x4.shape) 
x5 = tf.layers.max_pooling2d(x4, pool_size = [2,2], padding="same", strides = 2) 
print("Enc_layer6", x5.shape)
#x6 = tf.layers.conv2d(inputs= x5, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 
#print("Enc_layer7", x6.shape) 
#x7 = tf.layers.max_pooling2d(x6, pool_size = [2,2], padding="same", strides = 2) 
#print("Enc_layer8", x7.shape)
#x8 = tf.layers.conv2d(inputs= x7, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 
#print("Enc_layer9", x8.shape) 
#x9 = tf.layers.max_pooling2d(x8, pool_size = [2,2], padding="same", strides = 2) 
#print("Enc_layer10", x9.shape)
#x10 = tf.layers.conv2d(inputs= x9, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu) 
#print("Enc_layer11", x10.shape) 
#x11 = tf.layers.max_pooling2d(x10, pool_size = [2,2], padding="same", strides = 2) 
#print("Enc_layer12", x11.shape)
encoded_image = x5
print('Encoded_Image_Shape =',encoded_image.shape)


# let's check with cnv_2d Transpose structure
print("Decoder Arcitecture")
x_D1 = tf.contrib.layers.conv2d_transpose(encoded_image, num_outputs = 32, kernel_size = [3,3], stride=1,padding='SAME')
print("Dec_layer1", x_D1.shape)
#x_D2 = tf.image.resize_nearest_neighbor(x_D1, (2*8,2*8))
#print("Dec_layer2", x_D2.shape)
x_D3 = tf.contrib.layers.conv2d_transpose(x_D1, num_outputs = 32, kernel_size = [3,3], stride=1,padding='SAME')
print("Dec_layer3", x_D3.shape)
x_D4 = tf.image.resize_nearest_neighbor(x_D3, (2*16,2*16))
print("Dec_layer4", x_D4.shape)
x_D5 = tf.contrib.layers.conv2d_transpose(x_D4, num_outputs = 32, kernel_size = [3,3], stride=1,padding='SAME')
print("Dec_layer5", x_D5.shape)
x_D6 = tf.image.resize_nearest_neighbor(x_D5, (2*32,2*32))
print("Dec_layer6", x_D6.shape)
#x_D7 = tf.contrib.layers.conv2d_transpose(x_D6, num_outputs = 16, kernel_size = [3,3], stride=1,padding='SAME')
#print("Dec_layer7", x_D7.shape)
#x_D8 = tf.image.resize_nearest_neighbor(x_D7, (2*64,2*64))
#print("Dec_layer8", x_D8.shape)
#x_D9 = tf.contrib.layers.conv2d_transpose(x_D8, num_outputs = 32, kernel_size = [3,3], stride=1,padding='SAME')
#print("Dec_layer9", x_D9.shape) 
#x_D10 = tf.image.resize_nearest_neighbor(x_D9, (2*128,2*128))
#print("Dec_layer10", x_D10.shape)
#x_D11 = tf.contrib.layers.conv2d_transpose(x_D10, num_outputs = 32, kernel_size = [3,3], stride=1,padding='SAME')
#print("Dec_layer11", x_D11.shape) 
#x_D12 = tf.image.resize_nearest_neighbor(x_D9, (2*256,2*256))
#print("Dec_layer12", x_D12.shape)
decoded_image_ = tf.contrib.layers.conv2d_transpose(x_D6, num_outputs = 1, kernel_size = [3,3], stride=1,padding='SAME')
print("Final Layer in Decoder", decoded_image_.shape) 

decoded_image = tf.identity(decoded_image_, name = "Output")
print("another Final Layer in Decoder", decoded_image.shape)

### This block of code is for adaptive learning rate
def make_learning_rate_tensor(reduction_steps, learning_rates, global_step):
    assert len(reduction_steps) + 1 == len(learning_rates)
    if len(reduction_steps) == 1:
        return tf.cond(
            global_step < reduction_steps[0],
            lambda: learning_rates[0],
            lambda: learning_rates[1]
        )
    else:
        return tf.cond(
            global_step < reduction_steps[0],
            lambda: learning_rates[0],
            lambda: make_learning_rate_tensor(
                reduction_steps[1:],
                learning_rates[1:],
                global_step,)
            )
#

global_step = tf.train.get_or_create_global_step()
learning_rates = [0.01, 0.0000001]
#steps_per_epoch = 225
epochs_to_switch_at = [ 4000  ]
#epochs_to_switch_at = [x*steps_per_epoch for x in epochs_to_switch_at ]
Learning_Rate = make_learning_rate_tensor(epochs_to_switch_at , learning_rates, global_step)

#


mmse = tf.losses.mean_squared_error(decoded_image, X_Input)
optimizer = tf.train.AdadeltaOptimizer(learning_rate= Learning_Rate, rho=0.95, epsilon=1e-6).minimize(mmse)

init_op = tf.global_variables_initializer()
bal = 22000
#saver = tf.train.Saver()   ## I need to save the model right, that's why doing this operation # uncomment
#
with tf.Session() as sess:
    sess.run(init_op)
    Denoiser_loss = []
    for epoch in range(10):
#        for i in range(num_batches):
#            offset = (i * batch_size) % (X_train_noisy.shape[0] - batch_size)
#            batch_x_noisy = X_train_noisy[offset:(offset + batch_size)]
#            batch_x_orig = X_train_orig[offset:(offset + batch_size)]
#            _, u = sess.run([optimizer, mmse], feed_dict = {X_Input_Noise:batch_x_noisy.eval(), X_Input:batch_x_orig.eval()})
        _, u = sess.run([optimizer, mmse], feed_dict = {X_Input_Noise:X_train_noisy.eval(), X_Input: X_train_orig.eval()})
        print(epoch, u)
        Denoiser_loss.append(u)
    print(Denoiser_loss)
    
    
    any_image = X_train_noisy[20]    
    any_image = tf.reshape(any_image, [1,64,64,1])
    any_image_11 = tf.image.resize_images(any_image, [64,64])
    print("Shape after Resize", any_image_11.shape) 
    any_image_21 = tf.reshape(any_image_11, [64,64])
    print("Shape after Reshape", any_image_21.shape)
    any_vis_image = any_image_21.eval()
    Image.fromarray(any_vis_image.astype('uint8'), mode='L').show()
#
    output_any_image = sess.run(decoded_image,\
                 feed_dict={X_Input_Noise:any_image.eval()})
    print("output image", output_any_image.shape)
    print(tf.reduce_sum(output_any_image).eval())
    output_any_image_1 = tf.image.resize_images(output_any_image, [64,64])
    print("Shape after Resize", output_any_image_1.shape) 
    output_any_image_2 = tf.reshape(output_any_image_1, [64,64])
    print("Shape after Reshape", output_any_image_2.shape)
    vis_image = output_any_image_2.eval()
    Image.fromarray(vis_image.astype('uint8'), mode='L').show()    
#      
plt.plot(Denoiser_loss, '--r')
plt.title("Denosing Loss for Training Data")
plt.xlabel('No of Epochs')
plt.ylabel('Denoising Loss')        


