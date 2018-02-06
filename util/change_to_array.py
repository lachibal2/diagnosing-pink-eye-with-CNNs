import numpy
from PIL import Image
import os
import matplotlib.image as mpimg

def changeToArray(image, num):
    os.chdir('..')
    resizeSize = 128
    pic = Image.open('.\\' + image)
    new_width = resizeSize
    new_height = resizeSize
    pic = pic.resize((new_width, new_height), Image.ANTIALIAS)
    os.chdir('.\\changed_images')
    try:
        pic.save('.\\' + str(num) + '.png')
    except OSError:
        pic.save('.\\' + str(num) + '.jpg')
    finally:
        try:
            picArray = mpimg.imread(str(num) + '.png')#convert to array
        except FileNotFoundError:
            picArray = mpimg.imread(str(num) + '.jpg')

        return picArray
