import openslide
from PIL import Image

from math import ceil
import os
from os import listdir
from os.path import isfile, join, isdir
import glob



def get_image_paths(folder):
    image_paths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if join(folder, '.DS_Store') in image_paths:
        image_paths.remove(join(folder, '.DS_Store'))
    return image_paths


def get_subfolder_paths(folder):
    subfolder_paths = [join(folder, f) for f in listdir(folder) if (isdir(join(folder, f)) and '.DS_Store' not in f)]
    if join(folder, '.DS_Store') in subfolder_paths:
        subfolder_paths.remove(join(folder, '.DS_Store'))
    return subfolder_paths


def get_num_horizontal_positions(input_folder):
    horizontal_positions = []
    image_paths = get_image_paths(input_folder)
    for image_path in image_paths:
        x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[1])
        horizontal_positions.append(x_increment)
    return len(set(horizontal_positions))


def get_num_vertical_positions(input_folder):
    vertical_positions = []
    image_paths = get_image_paths(input_folder)
    for image_path in image_paths:
        x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[2])
        vertical_positions.append(x_increment)
    return len(set(vertical_positions))


def output_repieced_image(input_folder, output_image_path, window_size, compression_factor):
	
	Image.MAX_IMAGE_PIXELS = 1e10
	compressed_window_size = int(window_size / compression_factor)

	num_horizontal_positions = get_num_horizontal_positions(input_folder)
	num_vertical_positions = get_num_vertical_positions(input_folder)

	image_paths = get_image_paths(input_folder)
	images = map(Image.open, image_paths)
	widths, heights = zip(*(i.size for i in images))

	last_width = min(widths)
	last_height = min(heights)

	total_width = (num_horizontal_positions - 1)*compressed_window_size + last_width
	total_height = (num_vertical_positions - 1)*compressed_window_size + last_height

	new_im = Image.new('RGB', (total_width, total_height))

	for image_path in image_paths:

		x_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[1])
		y_increment = int(image_path.split('/')[-1].split('.')[0].split('_')[2])

		image = Image.open(image_path)
		new_im.paste(image, (compressed_window_size*x_increment, compressed_window_size*y_increment))

	new_im.save(output_image_path)



def main():


	window_size=224*20
	compression_factor=1.5

	images=glob.glob("input/*.svs")
	for image in images:
		img = openslide.OpenSlide(image)

	
		image_name=image.split('/')[-1][:-4]
		output_path=("tmp_patches")
	

		width, height = img.level_dimensions[0]

		increment_x = int(ceil(width / window_size))
		increment_y = int(ceil(height / window_size))

		print("converting", image_name, "with width", width, "and height", height)

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

				# save the image
			
				#output_subfolder = os.path.join(output_path, image_name)
			
				if not os.path.exists(output_path):
					os.makedirs(output_path)
				output_image_name = os.path.join(output_path,image_name+ '_' + str(incre_x) + '_' + str(incre_y) + '.tiff')
				print(output_image_name)
				patch_rgb.save(output_image_name, 'tiff')
			

		output_repieced_image(output_path, "output/"+image_name+".tiff", window_size, compression_factor)
		removing_files = glob.glob('tmp_patches/*.tiff')
		for i in removing_files:
			os.remove(i)
		print('#################################################')

if __name__ == '__main__':
	main()
