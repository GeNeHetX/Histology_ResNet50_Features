import openslide
from PIL import Image

from math import ceil, floor
import os
import glob

import pandas as pd

from warnings import simplefilter

from 0_Purple_Threshold import is_purple

from 0_Features import Calc_Features

from tensorflow.keras.applications.resnet50 import ResNet50


def main():

	print("\n->Initiating...\n\n->Loading The Model may take some time\n")
	model = ResNet50(weights='imagenet', include_top=False, pooling='max')

	print('\n->Processing WSI patches\n')
	simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

	padding=4

	images=glob.glob("Slides/*.svs")
	for image in images:
		img = openslide.OpenSlide(image)

		window_size=224*2
		image_name=image.split('/')[-1][:-4]
		output_path=("WSI_Features")
		compression_factor=2

		width, height = img.level_dimensions[0]

		increment_x = int(ceil(width / window_size))
		increment_y = int(ceil(height / window_size))

		print("\n->Processing", image_name, "with width", width, "and height", height, '\n')

		List_Purple=[]
		List_Xcor=[]
		List_Ycor=[]

		#random_X=floor(increment_x/60)
		#random_Y=floor(increment_y/60)

		feature_List=[]

		for incre_x in range(increment_x):  # read the image in patches
			for incre_y in range(increment_y):
				begin_x = window_size * incre_x
				end_x = min(width, begin_x + window_size)
				begin_y = window_size * incre_y
				end_y = min(height, begin_y + window_size)
				patch_width = end_x - begin_x
				patch_height = end_y - begin_y

				patch = img.read_region((begin_x, begin_y), 0, (patch_width, patch_height))
				patch.load()
				patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
				patch_rgb.paste(patch, mask=patch.split()[3])

				# compress the image
				patch_rgb = patch_rgb.resize((int(patch_rgb.size[0] / compression_factor), int(patch_rgb.size[1] / compression_factor)), Image.ANTIALIAS)

				# create folders
			
				output_subfolder = os.path.join(output_path, image_name)
			
				if not os.path.exists(output_subfolder):
					os.makedirs(output_subfolder)
				output_image_name = os.path.join(output_subfolder,image_name+ '_' + str(incre_x) + '_' + str(incre_y))
				print(output_image_name)
			
				# test color score
				try:
					num_purple, purple_thresh = is_purple(patch_rgb)
				except ValueError:
					num_purple=0
					purple_thresh=0
			
				List_Purple.append(num_purple)
				List_Xcor.append(incre_x)
				List_Ycor.append(incre_y)

				# calculate features

				if num_purple>purple_thresh:
					feature_List.append(Calc_Features(patch_rgb, model))

		# save files
		print("\n->Saving tsv files in corresponding subfolder in WSI_Features Folder\n")
		df=pd.DataFrame(list(zip(List_Xcor, List_Ycor, List_Purple)), columns=['X','Y', 'Purp_score'])
		df.to_csv(output_subfolder + "/" + image_name +".tsv", sep = "\t", index=False)

		df2=df.copy()
		df2.drop(df[df.Purp_score < purple_thresh+1].index, inplace=True)
		for n_f in range (2048):
			df2['F_'+str(n_f+1).zfill(padding)] = [item[n_f] for item in feature_List]
		df2.to_csv(output_subfolder + "/" + image_name +"_Features_ResNet50.tsv", sep=('\t'), index=False)
		print('\n->Saved\n#################################################################\n')

if __name__ == '__main__':
	main()
