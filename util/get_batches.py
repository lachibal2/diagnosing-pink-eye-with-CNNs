import numpy as np
import os
from .change_to_array import changeToArray

def createData(image_size=128):
    print('Loading Data. . .', end='   ')
    
    originalPath = os.getcwd()
    os.chdir('../training images')
    
    array = os.listdir()
    array.pop() #remove last element, the folder 'changed_images'

    images = []
    not_suitable = 0
    
    if 'changed_images' not in os.listdir():
        os.mkdir('.\\changed_images')
    
    os.chdir('.\\changed_images')
    
    for image in range(len(array)):
        #need to add an eye haar cascade to detect it and center it in the image
        try:
            current = changeToArray(array[image], image)
            if len(current) == image_size:
                images.append(current)
            else:
                not_suitable += 1
        except (OSError, FileNotFoundError) as e:
            not_suitable += 1

    os.chdir(originalPath)

    print('[DONE]')
            
    print('{} images accepted'.format(len(images)))
    print('{} images were not suitable'.format(not_suitable))

    return images

def getBatches(imageArray, batch_size, input_size, output_size, image_size=128):
    randomIndices = np.random.randint(len(imageArray), size=batch_size)
    x = np.zeros((batch_size, image_size, input_size))
    y = np.zeros((batch_size, output_size))

    for i in range(batch_size):
        currentIndex = randomIndices[i]
        x[i] = imageArray[currentIndex]
        y[i] = imageArray[currentIndex]

    return x, y
