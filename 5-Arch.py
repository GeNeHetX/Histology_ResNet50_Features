import pandas as pd
import numpy as np

import archetypes as arch




n_components=0.85
n_archetypes=30
print("-> Reading file...\n")
PCA_components=pd.read_csv('WSI_Features/PCA_components_'+str(n_components)+'_thresh.tsv', sep='\t')
PCA_Array=PCA_components.to_numpy()

aa = arch.AA(n_archetypes=n_archetypes)
print("\n-> Transforming to archetypes...\n")

PCA_arch = aa.fit_transform(PCA_Array)

name_list=[]
padding=4
for i in range(1, n_archetypes+1):
	name="Arch_"+str(i).zfill(padding)
	name_list.append(name)

PCA_arch_df= pd.DataFrame(data=PCA_arch, columns= name_list)


print("\n-> Saving file\n")

patch_ids=pd.read_csv('WSI_Features/Total_Features.tsv', sep='\t', usecols=['WSI_id','X', 'Y'])
PCA_arch_id= pd.concat([patch_ids, PCA_arch_df], axis=1)
PCA_arch_id.to_csv('WSI_Features/archetypes_'+str(n_archetypes)+'_at_'+str(n_components)+'_thresh.tsv', sep='\t', index=False)
