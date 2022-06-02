import pandas as pd
import os

import openslide
from PIL import Image
#from Functions import save_patch

def save_patch(image, begin_x, begin_y, folder):
	img = openslide.OpenSlide("Slides/"+image+".svs")

	window_size=224*2
	output_path=("Calculations/NMF_TumorCell/Top_Correlations_Patches/spearman/"+folder)
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

def main():
	Top_Mean_Corrs = pd.read_csv("Calculations/NMF_TumorCell/Top_Correlations/NMF_Mean_Top_Correlations_spearman.tsv", sep='\t')
	Top_Ratio_Corrs = pd.read_csv("Calculations/NMF_TumorCell/Top_Correlations/NMF_Ratio_Top_Correlations_spearman.tsv", sep='\t')


	for i in range(len(Top_Mean_Corrs)):
		Phenotype = str(Top_Mean_Corrs["Phenotype"][i])
		nComp = str(Top_Mean_Corrs["nComp"][i])
		NMF_id = str(Top_Mean_Corrs["NMF_id"][i])

		NMF_W_File = pd.read_csv("Calculations/NMF_TumorCell/nComp"+nComp+"/NMF_W_ALL_nComp"+nComp+".tsv", sep='\t')

		result = {}

		n_top_patches = 100
		top100 = NMF_W_File[NMF_id].nlargest(n_top_patches)
		print(top100)
		result[NMF_id] = []

		for i in top100.keys():
			result[NMF_id].append(NMF_W_File.iloc[i, :3].tolist())

		print(result)
	
		for cluster, value in result.items():
			for i in range(n_top_patches):
				WSI_id=(value[i][0])
				X=int((value[i][1]))
				Y=int((value[i][2]))

				folder=Phenotype+"/NMF_Mean"
				save_patch(WSI_id, X, Y, folder)

	for i in range(len(Top_Ratio_Corrs)):
		Phenotype = str(Top_Ratio_Corrs["Phenotype"][i])
		nComp = str(Top_Ratio_Corrs["nComp"][i])
		NMF_id = str(Top_Ratio_Corrs["NMF_id"][i])

		NMF_W_File = pd.read_csv("Calculations/NMF_TumorCell/nComp"+nComp+"/NMF_W_ALL_nComp"+nComp+".tsv", sep='\t')

		result = {}

		n_top_patches = 100
		top100 = NMF_W_File[NMF_id].nlargest(n_top_patches)
		print(top100)
		result[NMF_id] = []

		for i in top100.keys():
			result[NMF_id].append(NMF_W_File.iloc[i, :3].tolist())

		print(result)
		
		for cluster, value in result.items():
			for i in range(n_top_patches):
				WSI_id=(value[i][0])
				X=int((value[i][1]))
				Y=int((value[i][2]))

				folder=Phenotype+"/NMF_Ratio"
				save_patch(WSI_id, X, Y, folder)

if __name__ == '__main__':
	main()