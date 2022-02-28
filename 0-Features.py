from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np



def Calc_Features(patch, model):

	x = image.img_to_array(patch)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)

	return features[0]
