import openslide
from PIL import Image

from math import ceil, floor
import os
import glob

import pandas as pd



import Code_from_deepslide

import Purple_Threshold

#def output_jpeg_tiles(image_name, output_path):

images=glob.glob("Slides/*.svs")
for image in images:
	img = openslide.OpenSlide(image)

	window_size=224*2
	image_name=image.split('/')[-1][:-4]
	output_path=("patches")
	compression_factor=2

	width, height = img.level_dimensions[0]

	increment_x = int(ceil(width / window_size))
	increment_y = int(ceil(height / window_size))

	print("converting", image_name, "with width", width, "and height", height)

	List_Purple=[]
	
	List_Xcor=[]
	List_Ycor=[]

	random_X=floor(increment_x/10)
	random_Y=floor(increment_y/10)

	for incre_x in range(random_X, increment_x, random_X):  # read the image in patches
		for incre_y in range(random_Y, increment_y, random_Y):
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

			# save the image
			
			output_subfolder = os.path.join(output_path, image_name)
			
			if not os.path.exists(output_subfolder):
				os.makedirs(output_subfolder)
			output_image_name = os.path.join(output_subfolder,image_name+ '_' + str(incre_x) + '_' + str(incre_y) + '.jpg')
			print(output_image_name)
			if Code_from_deepslide.is_purple(patch_rgb):
				print("is purple")
				patch_rgb.save(output_image_name, 'jpeg')
			num_purple, purple_thresh = Purple_Threshold.is_purple(patch_rgb)
			
			List_Purple.append(num_purple)
			List_Xcor.append(incre_x)
			List_Ycor.append(incre_y)

	df=pd.DataFrame(list(zip(List_Xcor, List_Ycor, List_Purple)), columns=['X','Y', 'Purp_score'])
	df.to_csv(output_subfolder + "/" + image_name +"_Purple_JPEGs.tsv", sep = "\t", index=False)
