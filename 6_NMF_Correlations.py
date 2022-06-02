import pandas as pd
import numpy as np
import os

from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def main():
	phenotype_raw=pd.read_csv("molecular/BJ100_MolcharacFitted.tsv", sep='\t')
	EXCLUDE = ['P20AG05411', '13AG04004-15', '13AG05411-15', '549407-21', '549407.25', '552252-08']


	n_components_List=list(range(2,201))

	for n_components in n_components_List:
		print("---> n_components = ", n_components)

		output_subfolder= 'Calculations/NMF_TumorCell/nComp'+str(n_components)

		if not os.path.exists(output_subfolder+"/Correlations"):
			os.makedirs(output_subfolder+"/Correlations")


		NMF_Ratio=pd.read_csv(output_subfolder+"/NMF_W_ALL_nComp"+str(n_components)+"_Label_Ratio.tsv", sep='\t')
		NMF_Mean=pd.read_csv(output_subfolder+"/NMF_W_ALL_nComp"+str(n_components)+"_Mean.tsv", sep='\t')


		phenotype=phenotype_raw[phenotype_raw.WSI_id.isin(EXCLUDE) == False]
		NMF_Mean=NMF_Mean[NMF_Mean.WSI_id.isin(EXCLUDE) == False]
		NMF_Ratio=NMF_Ratio[NMF_Ratio.WSI_id.isin(EXCLUDE) == False]

		###---->Make All Values Positive

		for column in phenotype.columns[1:]:
			min = phenotype[column].min()
			phenotype[column]=phenotype[column] - min

	##################### Pairwise correlation ###############################
		corr_ratio=pd.concat([NMF_Ratio, phenotype], axis=1, keys=['NMF_Ratio', 'phenotype']).corr('spearman').loc['phenotype', 'NMF_Ratio']
		corr_mean=pd.concat([NMF_Mean, phenotype], axis=1, keys=['NMF_Mean', 'phenotype']).corr('spearman').loc['phenotype', 'NMF_Mean']

		#print("Pairwise Correlation with NMF Ratio: \n", corr_ratio)
		#print("Pairwise Correlation with NMF Mean: \n", corr_mean)

		corr_ratio.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Ratio_Correlation_spearman.tsv", sep='\t')
		corr_mean.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Mean_Correlation_spearman.tsv", sep='\t')

	##################### NNLS correlation ###############################

		NMF_Mean=NMF_Mean.drop(['WSI_id'], axis=1)
		NMF_Ratio=NMF_Ratio.drop(['WSI_id'], axis=1)
		phenotype=phenotype.drop(['WSI_id'], axis=1)

		NMF_id=list(NMF_Mean.columns)
		phenotype_id=list(phenotype.columns)


	###---->NNLS on NMF Means
		"""
		n = phenotype.shape[1]
		Means_Coef_nnls = np.zeros((NMF_Mean.shape[1], n))
		Means_rnorm = np.zeros((1, n))
	
		for i in range(n):
			Means_Coef_nnls[:,i], Means_rnorm[:,i] =  nnls(NMF_Mean.to_numpy(), phenotype.to_numpy()[:,i], maxiter=10000)
	
		Means_Coef_nnls=pd.DataFrame(Means_Coef_nnls, columns=phenotype_id, index=NMF_id).transpose()
		#print(Means_Coef_nnls)
		Means_rnorm=pd.DataFrame(Means_rnorm, columns=phenotype_id, index=['rnorm'])
		#print(Means_rnorm)
	
		Means_Coef_nnls.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Mean_nnls_Coef.tsv", sep='\t')
		Means_rnorm.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Mean_nnls_rnorm.tsv", sep='\t')
		"""


		reg_nnls = LinearRegression(positive=True)
		y_pred_nnls = pd.DataFrame(reg_nnls.fit(NMF_Mean, phenotype).predict(NMF_Mean), columns=phenotype_id)

		r2_List=[]
		for i in range(phenotype.shape[1]):
			r2_score_nnls = r2_score(phenotype.iloc[:,i], y_pred_nnls.iloc[:,i])
			r2_List.append(r2_score_nnls)
		r2_List=pd.DataFrame(r2_List, columns=['R_Square'], index=phenotype_id).transpose()
		print("NNLS R2 scores with NMF_Mean\n", r2_List)
		r2_List.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Mean_nnls_R2.tsv", sep='\t')
		#print(pd.DataFrame(reg_nnls.coef_, columns=NMF_id, index=phenotype_id))

	###---->NNLS on NMF Ratios

		"""
		n = phenotype.shape[1]
		Ratios_Coef_nnls = np.zeros((NMF_Ratio.shape[1], n))
		Ratios_rnorm = np.zeros((1, n))
	
		for i in range(n):
			Ratios_Coef_nnls[:,i], Ratios_rnorm[:,i] =  nnls(NMF_Ratio.to_numpy(), phenotype.to_numpy()[:,i], maxiter=10000)
	
		Ratios_Coef_nnls=pd.DataFrame(Ratios_Coef_nnls, columns=phenotype_id, index=NMF_id).transpose()
		#print(Ratios_Coef_nnls)
		Ratios_rnorm=pd.DataFrame(Ratios_rnorm, columns=phenotype_id, index=['rnorm'])
		#print(Ratios_rnorm)
	
		Ratios_Coef_nnls.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Ratio_nnls_Coef.tsv", sep='\t')
		Ratios_rnorm.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"._Ratio_nnls_rnorm.tsv", sep='\t')
		"""

		reg_nnls = LinearRegression(positive=True)
		y_pred_nnls = pd.DataFrame(reg_nnls.fit(NMF_Ratio, phenotype).predict(NMF_Ratio), columns=phenotype_id)
		r2_List=[]
		for i in range(phenotype.shape[1]):
			r2_score_nnls = r2_score(phenotype.iloc[:,i], y_pred_nnls.iloc[:,i])
			r2_List.append(r2_score_nnls)
		r2_List=pd.DataFrame(r2_List, columns=['R_Square'], index=phenotype_id).transpose()
		print("NNLS R2 scores with NMF_Ratio\n", r2_List)
		r2_List.to_csv(output_subfolder+"/Correlations/NMF_W_ALL_Comp"+str(n_components)+"_Ratio_nnls_R2.tsv", sep='\t')

if __name__ == '__main__':
	main()