import pandas as pd
import numpy as np

def main():
	n_components_List=list(range(2,201))

	for n_components in n_components_List:
		output_subfolder= 'Calculations/NMF_TumorCell/nComp'+str(n_components)

		print("---> n_components = ", n_components)
		NMFs=pd.read_csv(output_subfolder+"/NMF_W_ALL_nComp"+str(n_components)+".tsv", sep='\t')
		WSI_id = NMFs['WSI_id']
		NMFs=NMFs.drop(['X','Y'], axis=1)

		NMF_mean=NMFs.groupby(['WSI_id']).mean()

		labels = pd.DataFrame()
		labels['WSI_id']=WSI_id
		labels['label'] = (NMFs.iloc[:, 1:]).idxmax(axis=1)

		label_ratio=pd.DataFrame(columns=NMFs.columns)
		label_ratio['WSI_id']=WSI_id

		for i in range(len(labels)):
			label = labels.at[i,'label']
			label_ratio.at[i,label]=1

		label_ratio=label_ratio.fillna(0)
		grp_n=label_ratio.groupby(['WSI_id']).count()
		label_ratio = label_ratio.groupby(['WSI_id']).sum()
		label_ratio=label_ratio/grp_n

		print("->Saving Label Ratio File... ")
		print(label_ratio)
		label_ratio.to_csv(output_subfolder+'/NMF_W_ALL_nComp'+str(n_components)+'_Label_Ratio.tsv', sep='\t')

		print("->Saving NMF Mean File... ")
		print(NMF_mean)
		NMF_mean.to_csv(output_subfolder+'/NMF_W_ALL_nComp'+str(n_components)+'_Mean.tsv', sep='\t')

if __name__ == '__main__':
	main()