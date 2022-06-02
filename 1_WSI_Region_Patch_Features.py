import openslide
from PIL import Image

from math import ceil, floor
import os
import glob
import xml.etree.ElementTree as ET

import pandas as pd

from warnings import simplefilter

from Functions import is_tissue, Calc_Features, patching_whole_slide

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

        List_tissue=[]
        List_Xcor=[]
        List_Ycor=[]
        feature_List=[]
        
        if not os.path.exists('Slides/'+image_name+'.xml'):
            print('Careful: Slide '+image_name+' Has No Annotation, Patching WHOLE SLIDE !\n')
            patching_whole_slide(image, model, window_size, compression_factor)
            continue

        w, h = img.level_dimensions[0]
        print("\n->Processing", image_name, "with width", w, "and height", h, '\n')

        tree = ET.parse('Slides/'+image_name+'.xml')
        root = tree.getroot()

        for region in root.findall('./Annotation/Regions/Region'):
            X, Y = [], []
            Id =  region.attrib["Id"]
            print(f"\nRegion Id: {Id}")

            ROA =  region.attrib["NegativeROA"]
            if ROA == "1":
                print("Negative Region\n")
            else:

                Vertices = region.findall("./Vertices/Vertex")
                for vertex in Vertices:
                    X.append(float(vertex.attrib['X']))
                    Y.append(float(vertex.attrib['Y']))
  
                Xmin, Xmax, Ymin, Ymax = int(min(X)), int(max(X)), int(min(Y)), int(max(Y))

                crop_begin_x, crop_end_x = Xmin, Xmax
                crop_begin_y, crop_end_y = Ymin, Ymax
                crop_patch_w = crop_end_x - crop_begin_x
                crop_patch_h = crop_end_y - crop_begin_y 

                increment_x = int(ceil(crop_patch_w / window_size))
                increment_y = int(ceil(crop_patch_h / window_size))

                print("\n->Processing Region with width", crop_patch_w, "and height", crop_patch_h, '\n')

                for incre_x in range(increment_x):  # read the image in patches
                    for incre_y in range(increment_y):
                        begin_x = window_size * incre_x + crop_begin_x
                        end_x = min(crop_end_x, begin_x + window_size)
                        begin_y = window_size * incre_y + crop_begin_y
                        end_y = min(crop_end_y, begin_y + window_size)
                        patch_width = end_x - begin_x
                        patch_height = end_y - begin_y

                        patch = img.read_region((begin_x, begin_y), 0, (patch_width, patch_height))
                        patch.load()
                        patch_rgb = Image.new("RGB", patch.size, (255, 255, 255))
                        patch_rgb.paste(patch, mask=patch.split()[3])

                        # compress the image
                        try:
                            patch_rgb = patch_rgb.resize((int(patch_rgb.size[0] / compression_factor),
                                int(patch_rgb.size[1] / compression_factor)), Image.ANTIALIAS)
                        except:
                            continue

                        # create folders
            
                        output_subfolder = os.path.join(output_path, image_name)
            
                        if not os.path.exists(output_subfolder):
                            os.makedirs(output_subfolder)
                        #output_image_name = os.path.join(output_subfolder,image_name+ '_' + str(begin_x) + '_' + str(begin_y))
                        print(image_name+ '_' + str(begin_x) + '_' + str(begin_y))
            
                        # test color score
                        try:
                            num_tissue, tissue_thresh = is_tissue(patch_rgb)
                        except ValueError:
                            num_tissue=0
                            tissue_thresh=0.25
            
                        List_tissue.append(num_tissue)
                        List_Xcor.append(begin_x)
                        List_Ycor.append(begin_y)

                        # calculate features

                        if num_tissue >= tissue_thresh:
                            feature_List.append(Calc_Features(patch_rgb, model))

        # save files
        print("\n->Saving tsv files in corresponding subfolder in WSI_Features Folder\n")
        df=pd.DataFrame(list(zip(List_Xcor, List_Ycor, List_tissue)), 
            columns=['X', 'Y', 'Tissue_score'])

        df2=df.copy()
        df2.drop(df[df.Tissue_score < tissue_thresh].index, inplace=True)
        for n_f in range (2048):
            df2['F_'+str(n_f+1).zfill(padding)] = [item[n_f] for item in feature_List]

        df = df.drop_duplicates(keep='first')
        df.to_csv(output_subfolder + "/" + image_name +".tsv", sep = "\t", index=False)

        df2 = df2.drop_duplicates(keep='first')            
        df2.to_csv(output_subfolder + "/" + image_name +"_Features_ResNet50.tsv", sep=('\t'), index=False)
        print('\n->Saved\n#################################################################\n')

if __name__ == '__main__':
    main()
