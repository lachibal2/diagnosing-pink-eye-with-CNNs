import numpy as np
import os
from .change_to_array import changeToArray

def createData(image_size=128):
    print('Loading Data. . .', end='   ')
    
    originalPath = os.getcwd()
    os.chdir(originalPath + '\\training images')

    images = []
    labels = []
    not_suitable = 0
    
    #tries to create the directory, if it does not exist
    try:
        os.mkdir(originalPath + '\\changed_images')

    except FileExistsError as e:
        pass

    array = os.listdir()
    
    os.chdir(originalPath + '\\training images')
    
    for image in range(len(array)):

        try:
            current = changeToArray(array[image], image)
            
            if len(current) == image_size:
                images.append(current)
                
                if '_P_' in array[image]:
                    labels.append(1)

                elif '_NP_' in array[image]:
                    labels.append(0)

                else:
                    not_suitable += 1
                    images.pop()
            
            else:
                not_suitable += 1

        except (OSError, FileNotFoundError) as e:
            not_suitable += 1

    os.chdir(originalPath)

    print('[DONE]')
            
    print('{} images accepted'.format(len(images)))
    print('{} images were not suitable'.format(not_suitable))

    return {"images":images, "labels": labels}

def getBatches(imageArray, batch_size, input_size, output_size, image_size=128):
    randomIndices = np.random.randint(len(imageArray), size=batch_size)
    x = np.zeros((batch_size, image_size, input_size))
    y = np.zeros((batch_size, output_size))

    for i in range(batch_size):
        currentIndex = randomIndices[i]
        x[i] = imageArray[currentIndex]
        y[i] = imageArray[currentIndex]

    return x, y
