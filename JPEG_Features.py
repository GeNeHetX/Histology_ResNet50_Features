from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import glob
import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


model = ResNet50(weights='imagenet', include_top=False, pooling='max')



c=0
c1=0
padding=4

folders=glob.glob("patches/*")
for folder in folders:

	patches=glob.glob(folder + "/*.jpg")
	feature_List=[]
	for patch in patches:


		img_path = patch
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)

		features = model.predict(x)

		c=c+1
		print(c)

		feature_List.append(features[0])

		#print(len(feature_List[c]))
		#c1=c1+1

	pd.set_option("display.max_rows", None, "display.max_columns", None)
	df = pd.read_csv(patch.split('_')[0]+".tsv", sep='\t')

	for n_f in range (2048):

		df['F_'+str(n_f+1).zfill(padding)] = [item[n_f] for item in feature_List]

	#for n in range (0,len(feature_List)):
		#print('line number   ', n, 'feature lenght is : ' , len(df['FEATURES'][n]))
	
	df.to_csv("patches/"+patch.split('/')[1]+"/"+patch.split('/')[1]+"_Features_ResNet50_JPEGs.tsv", sep=('\t'), index=False)
	#print(df)
	print('#################################################')