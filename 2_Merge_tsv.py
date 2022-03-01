import os
import pandas as pd
import glob

def main():
	folders = glob.glob("WSI_Features/*/")

	li=[]
	print('-> Initiating: ')
	c=0
	for folder in folders:
		print("File number: ", c, " Out of: ", len(folders))
		selected_csv = (folder.split('/')[1]+'_Features_ResNet50.tsv')
		filename = os.path.join(folder, selected_csv)

		df = pd.read_csv(filename, index_col=None, header=0, sep='\t')
		count_row = df.shape[0]

		id_list=[]
		for i in range(count_row):
			id_list.append(folder.split('/')[1])
	
		df.insert(0, "WSI_id", id_list, True)

		li.append(df)

	print("-> Making final file: Total_Features.tsv")
	frame = pd.concat(li, axis=0, ignore_index=True)
	frame.to_csv('WSI_Features/Total_Features.tsv',index=False, sep='\t')
	print("-> Saved")

if __name__ == '__main__':
	main()
