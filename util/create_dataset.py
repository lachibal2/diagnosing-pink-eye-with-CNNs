import numpy as np

def get_batch(image_list, batch_size):
	#generator, yields next *batch_size* elements in the image_list
	lower = 0
	for i in image_list:
		yield image_list[lower: lower + batch_size]
		lower += batch_size