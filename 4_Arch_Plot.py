import pandas as pd
import numpy as np

import archetypes as arch
import matplotlib.pyplot as plt




n_components=0.85
print('-> Reading file...\nPCA_components_'+str(n_components)+'_thresh.tsv\n')
PCA_components=pd.read_csv('WSI_Features/PCA_components_'+str(n_components)+'_thresh.tsv', sep='\t')
PCA_Array=PCA_components.to_numpy()

print("\n-> Calculating errors...\n")
rsj=[]
max_n_arch=300

for j in range(1,max_n_arch+1):
    rsi=[]
    print("-------------N_Arch= ", j)
    for i in range(1,10):
        rsi.append(arch.AA(n_archetypes=j, n_init=1, max_iter=10).fit(PCA_Array).rss_)
        print("Iteration= ", i)
    rsj.append(rsi)


print("\n-> Making boxplot...\n")

#   plot
fig = plt.figure(figsize =(20, 20))
 
# box plot
plt.boxplot(rsj)
 
# show plot
#plt.show()
#plt.savefig("arch_boxplot.png")

print("\n-> Making plot...\n")

rso=[min(o) for o in rsj]
#for o in range(1,max_n_arch+1):
 #   rso.append(arch.AA(n_archetypes=j, n_init=1, max_iter=10).fit(PCA_Array).rss_)
plot_list=[i for i in range (1, len(rso)+1)]
plt.plot(plot_list,rso)
plt.savefig("figures/arch_plot.png")

