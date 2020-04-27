import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimg

def changeToArray(image, num):
    ORIGINAL_DIR = os.getcwd()
    resizeSize = 128
    cdir = os.getcwd()
    
    pic = Image.open(cdir + "\\" + image)
    new_width = resizeSize
    new_height = resizeSize
    pic = pic.resize((new_width, new_height), Image.ANTIALIAS)
   
    os.chdir(cdir + '\\..\\changed_images')

    cdir = os.getcwd()

    try:
        pic.save(cdir + '\\' + str(num) + '.png')
    
    except OSError:
        pic.save(cdir + '\\' + str(num) + '.jpg')
    
    finally:
        try:
            picArray = np.array(mpimg.imread(str(num) + '.png'))#convert to array
        
        except FileNotFoundError:
            picArray = np.array(mpimg.imread(str(num) + '.jpg'))

        os.chdir(ORIGINAL_DIR)

        return picArray
