from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

import openslide
from PIL import Image

from math import ceil, floor
import os
import glob

import pandas as pd

from warnings import simplefilter


def Calc_Features(patch, model):

	x = image.img_to_array(patch)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)

	return features[0]
#####################################################################################
def is_tissue(crop, tissue_threshold = 0.25) -> bool:


    gray = crop.convert('L')
    bw = gray.point(lambda x: 0 if x<220 else 1, 'F')
    arr = np.array(np.asarray(bw))
    num_white = np.average(bw)
                    
    num_tissue = float(format(1-num_white, ".2f"))

                        
    return num_tissue, tissue_threshold

####################################################################################
def patching_whole_slide(image, model, window_size, compression_factor):

    print("\n->Initiating...\n\n->Patching Whole Slide may take some time\n")

    print('\n->Processing WSI patches\n')
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    padding=4

    img = openslide.OpenSlide(image)

    image_name=image.split('/')[-1][:-4]
    output_path=("WSI_Features")
    
    width, height = img.level_dimensions[0]

    increment_x = int(ceil(width / window_size))
    increment_y = int(ceil(height / window_size))

    print("\n->Processing", image_name, "with width", width, "and height", height, '\n')

    List_tissue=[]
    List_Xcor=[]
    List_Ycor=[]

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
            print(image_name+ '_' + str(begin_x) + '_' + str(begin_y))
            
            # test color score
            try:
                num_tissue, tissue_thresh = is_tissue(patch_rgb)
            except ValueError:
                num_tissue=0
                tissue_thresh=0.25
            
            List_tissue.append(num_tissue)
            List_Xcor.append(incre_x)
            List_Ycor.append(incre_y)

            # calculate features

            if num_tissue >= tissue_thresh:
                feature_List.append(Calc_Features(patch_rgb, model))

    # save files
    print("\n->Saving tsv files in corresponding subfolder in WSI_Features Folder\n")
    df=pd.DataFrame(list(zip(List_Xcor, List_Ycor, List_tissue)), columns=['X','Y', 'Tissue_score'])
    df.to_csv(output_subfolder + "/" + image_name +".tsv", sep = "\t", index=False)

    df2=df.copy()
    df2.drop(df[df.Tissue_score < tissue_thresh].index, inplace=True)
    for n_f in range (2048):
        df2['F_'+str(n_f+1).zfill(padding)] = [item[n_f] for item in feature_List]
    df2.to_csv(output_subfolder + "/" + image_name +"_Features_ResNet50.tsv", sep=('\t'), index=False)
    print('\n->Saved\n#################################################################\n')
############################################################################################################
def save_patch(image, begin_x, begin_y, folder):
	img = openslide.OpenSlide("Slides/"+image+".svs")

	window_size=224*2
	output_path=("Top_Patches"+folder)
	compression_factor=2

	width, height = img.level_dimensions[0]

	end_x = min(width, begin_x + window_size)
	end_y = min(height, begin_y + window_size)
	patch_width = end_x - begin_x
	patch_height = end_y - begin_y

	patch = img.read_region((begin_x, begin_y), 0, (patch_width, patch_height))
	patch.load()
	patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
	patch_rgb.paste(patch, mask=patch.split()[3])

	# save the image
			
			
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	output_image_name = os.path.join(output_path,image+ '_' + str(begin_x) + '_' + str(begin_y) + '.jpg')
	print(output_image_name)
	
	patch_rgb.save(output_image_name, 'jpeg')