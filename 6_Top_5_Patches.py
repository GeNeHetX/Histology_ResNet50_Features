import pandas as pd
import openslide
from PIL import Image
import os



def save_patch(image, incre_x, incre_y, arch):
	img = openslide.OpenSlide("Slides/"+image+".svs")

	window_size=224*2
	image_name=image.split('/')[-1][:-4]
	output_path=("Top_Patches/"+str(arch))
	compression_factor=2

	width, height = img.level_dimensions[0]


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
	#patch_rgb = patch_rgb.resize((int(patch_rgb.size[0] / compression_factor), int(patch_rgb.size[1] / compression_factor)), Image.ANTIALIAS)

	# save the image
			
			
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	output_image_name = os.path.join(output_path,image_name+ '_' + str(incre_x) + '_' + str(incre_y) + '.jpg')
	print(output_image_name)
	
	patch_rgb.save(output_image_name, 'jpeg')


def main():
	df = pd.read_csv('WSI_Features/archetypes_'+str(n_archetypes)+'_at_'+str(n_components)+'_thresh.tsv', sep='\t')


	result = {}

	for column in df.columns[3:]:

	    top5 = df[column].nlargest(5)
	    result[column] = []

	    for i in top5.keys():
	        result[column].append(df.iloc[i, :3].tolist())

"""   optional   """
	#t = pd.DataFrame(result)

	#print(t)
	#print(t['Arch_0001'])

	for arch, value in result.items():
		for i in range(5):
			WSI_id=(value[i][0])
			X=(value[i][1])
			Y=(value[i][2])

			save_patch(WSI_id, X, Y, arch)

if __name__ == '__main__':
	main()
