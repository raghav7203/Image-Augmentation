# All imports
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from math import floor, ceil, pi
import warnings
warnings.filterwarnings('ignore')

# Helping functions

def get_images(path):
    data = os.listdir(path)
    return data

def show_one_images(img):
    fig=plt.figure(figsize=(8,8))
    plt.imshow(img)
    plt.show()

def show_two_images(original_image , new_image , cmap=None):
    fig=plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,1)
    plt.imshow(original_image)
    fig.add_subplot(1,2,2)
    plt.imshow(new_image,cmap=cmap)
    plt.show()

# Collecting images

images = get_images('./images/')
print(images)
for i in range(0,len(images)):
 img=mpimg.imread('./images/' + images[i])
 plt.figure(figsize=(6,6))
 plt.imshow(img)
 plt.show()
 from skimage.util import random_noise
 random_noise_image = random_noise(img)
 print(random_noise_image.shape)
 show_two_images(img,random_noise_image)

 import imageio
 imageio.imwrite('./Generated_Images/random_noise_image_'+str(i)+'.png' , random_noise_image)

'''
# 1 Rescale Images

from skimage.transform import rescale
rescaled_image = rescale(img , 0.5 ,mode='constant')
show_two_images(img,rescaled_image[:,:,0])
#show_two_images(img,rescaled_image[:,:,0]) # here color is changing but if i pass only rescale_image then type error is there


# 2 Adding random noise

from skimage.util import random_noise
random_noise_image = random_noise(img)
print(random_noise_image.shape)
show_two_images(img,random_noise_image)

# 3 Coloured to Grey Scale

from skimage.color import rgb2grey
gray_image = rgb2grey(img)

show_two_images(img,gray_image,cmap="gray") # cmap used to highlight intensities

# 4 Image Color Conversion

invert_color_image = np.invert(img)

show_two_images(img,invert_color_image)

# 5 Rotate Image

from skimage.transform import rotate

angle1 = 45
angle2 = 75
angle3 = -45
angle4 = -45

rotated_image1 = rotate(img,angle1)
rotated_image2 = rotate(img,angle2)
rotated_image3 = rotate(img,angle3)
rotated_image4 = rotate(img,angle4)

show_two_images(img,rotated_image1)
show_two_images(img,rotated_image2)
show_two_images(img,rotated_image3)
show_two_images(img,rotated_image4)


# 6 Rescale intensity

from skimage import exposure
v_min , v_max = np.percentile(img , (0.2,70.8))
contrast_image = exposure.rescale_intensity(img , in_range=(v_min , v_max))

show_two_images(img,contrast_image)

# 7 Gamma Correction

gamma_image = exposure.adjust_gamma(img , gamma=0.4 , gain=0.8)

show_two_images(img,gamma_image)


# 8 Horizontal Flip
hor_flip_image = img[: , ::-1]

show_two_images(img,hor_flip_image)

# 9 Vertical Flip
vert_flip_image = img[::-1 , :]

show_two_images(img,vert_flip_image)

# 10 Red Blur
red_blur_image = img[:,:,0]

show_two_images(img,red_blur_image,cmap='Reds')

# 11 Green Blur
green_blur_image = img[:,:,1]

show_two_images(img,green_blur_image,cmap='Greens')

# 12 Blue Blur
blue_image_blur = img[:,:,2]

show_two_images(img,blue_image_blur,cmap='Blues')

# 13 Sunny Weather
def add_sunny_weather(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    random_brightness_coefficient = np.random.uniform()+0.5
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient 
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    return image_RGB

bright_image = add_sunny_weather(img)

show_two_images(img,bright_image)

# 14 Shady Weather
def add_shady_weather(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    random_brightness_coefficient = np.random.uniform()+0.3
    image_HLS[:,:,1] = image_HLS[:,:,1]*random_brightness_coefficient 
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    return image_RGB

shady_image = add_shady_weather(img)

show_two_images(img,shady_image)

# 15 Snowy Weather With Sunny Conditions
def add_snow_sunny_weather(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 2.5 
    snow_point=140 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    return image_RGB

snow_image = add_snow_sunny_weather(img)

show_two_images(img,snow_image)

# 16  Snowy Weather With Shady Conditions
def add_snow_shady_weather(image):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 0.5
    snow_point=140 ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    return image_RGB

snow_image = add_snow_shady_weather(img)

show_two_images(img,snow_image)

# 17 Rain Weather
def add_water_drops(imshape,slant,drop_length):
    drops=[]
    for i in range(1500): 
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops

def add_rain(image):
    
    imshape = image.shape
    slant_extreme=10
    slant= np.random.randint(-slant_extreme,slant_extreme) 
    drop_length=20
    drop_width=2
    drop_color=(200,200,200) 
    rain_drops= add_water_drops(imshape,slant,drop_length)
    
    for rain_drop in rain_drops:
        cv2.line(image,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image,(7,7))
    
    brightness_coefficient = 0.7 
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient 
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) 
    return image_RGB

rain_image = add_rain(img)

show_one_images(rain_image)


# Saving The Image
import imageio
imageio.imwrite('./Generated_Images/random_noise_image.png' , random_noise_image)
#imsave('./Generated_Images/random_noise_image.png' , random_noise_image)
#imsave('./Generated_Images/gray_image.png' , gray_image)
'''
