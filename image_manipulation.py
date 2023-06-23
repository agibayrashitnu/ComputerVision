#credit: Juan Carlos Niebles and Ranjay Krishna

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    # Use skimage io.imread
    
    image = io.imread(image_path)
    out = image
    
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """


    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    x_p = image
    x_n = 0.5*(x_p^2)/255
    out = x_n
    
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width)
    """

    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    
    gray_scale = image 
    gray_scale[:] = np.sum(gray_scale, axis = -1, keepdims = True)/5
    out = gray_scale

    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out

def rgb_decomposition(image, channel):
    """ Return image with the rgb channel specified
    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: string specifying the channel
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
   
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    if channel == "R":
        red = image.copy()
        red[:, :, 1] = 0
        red[:, :, 2] = 0
        out = red
        
    elif channel == "G":
        green = image.copy()
        green[:, :, 0] = 0
        green[:, :, 2] = 0
        out = green
        
    elif channel == "B":
        blue = image.copy()
        blue[:, :, 0] = 0
        blue[:, :, 1] = 0
        out = blue
        
    else:
        pass
        
    ######################################
    #        END OF YOUR CODE            #
    ######################################
    
    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 including only
    the specified channels for each image
    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: string specifying channel used for image1
        channel2: string specifying channel used for image2
    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    
    #####################################
    #       START YOUR CODE HERE        #
    #####################################
    height1, width1, third = image1.shape # 552 980 3
    height2, width2, third = image2.shape # 552 980 3
    
    image1_first_half_width = width1 // 2 #490
    image2_second_half_width = width2 // 2 #490
    
    left_half = image1[:, :image1_first_half_width]
    right_half = image2[:, image1_first_half_width: ]
    
    channel = ["R","G","B"]
    
    if channel1 in channel:
        out1 = rgb_decomposition(left_half, channel1)
    if channel2 in channel:
        out2 = rgb_decomposition(right_half, channel2)
    
    out = np.concatenate((out1,out2), axis=1)
    
    ######################################
    #        END OF YOUR CODE            #
    ######################################

    return out