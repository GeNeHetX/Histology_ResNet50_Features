import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
	print('-> Intitiating: ')
	df = pd.read_csv("WSI_Features/Total_Features.tsv", sep='\t')
	df=df.drop(['WSI_id','X','Y','Purp_score'], axis=1)


	X_std = StandardScaler().fit_transform(df)




	n_components=0.85
	print("-> Making PCA at threshold: ", n_components)
	pca = PCA(n_components=n_components)


	principalComponents = pca.fit_transform(X_std)
	print("\nNumber of components at ", n_components, "threshold is: ", len(pca.components_))

	name_list=[]
	padding=4
	for i in range(1, len(pca.components_)+1):
		name="C_"+str(i).zfill(padding)
		name_list.append(name)


	PCA_components = pd.DataFrame(principalComponents, columns= name_list)
	PCA_components.to_csv('WSI_Features/PCA_components_'+str(n_components)+'_thresh.tsv', sep='\t', index=False)

if __name__ == '__main__':
	main()

