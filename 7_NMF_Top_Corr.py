import pathlib
import pandas as pd
import numpy as np
import os

def main():
    init = pd.read_csv("./Calculations/NMF_TumorCell/nComp2/Correlations/NMF_W_ALL_Comp2_Mean_Correlation_spearman.tsv", sep="\t")
    init = init.to_numpy()
    col, row = init.shape
    minimum = -100

    IDs = init[:, 0]

    max_c = minimum * np.ones(col)
    n_com = np.zeros(col)
    max_is_in_col = -1*np.ones(col)

    for path in pathlib.Path("./Calculations/NMF_TumorCell").iterdir():
        try:
            nComp = path.name.split("nComp")[1]
        except:
            continue
        path=path/'Correlations'

        for file in path.iterdir():
            fileName = file.name
            if "Mean_Correlation_spearman.tsv" in fileName and fileName.endswith(".tsv"):
                data = pd.read_csv(file, sep="\t")
                t_col, t_row = data.shape

                columnsNames = data.columns[1:]
                colNamesIndex = np.arange(1, t_row)

                data = data.to_numpy()

                new_max = np.amax(data[:, 1:], axis=1)

                coll = np.array([new_max == data[:, i] for i in range(1, t_row)]).T
                new_col = np.array([colNamesIndex[i][0] for i in coll])
                max_is_in_col[new_max > max_c] = new_col[new_max > max_c]

                n_com[new_max > max_c] = nComp
                max_c[new_max > max_c] = new_max[new_max > max_c]



    result_mean = np.c_[IDs, max_c, max_is_in_col, n_com]
    result_mean = pd.DataFrame(result_mean, columns=["Phenotype", "Max_corr", "NMF_id", "nComp"])
    for i in range(len(result_mean['NMF_id'])):
        result_mean['NMF_id'][i] = "NMF_"+str(int(result_mean['NMF_id'][i])).zfill(4)
    result_mean = result_mean.astype({"Phenotype": str, "Max_corr": float, "NMF_id": str,"nComp": int})

    print("\n----> Mean\n", result_mean)

    max_c = minimum * np.ones(col)
    n_com = np.zeros(col)
    max_is_in_col = -1*np.ones(col)

    for path in pathlib.Path("./Calculations/NMF_TumorCell").iterdir():
        try:
            nComp = path.name.split("nComp")[1]
        except:
            continue
        path=path/'Correlations'

        for file in path.iterdir():
            fileName = file.name
            if "Ratio_Correlation_spearman.tsv" in fileName and fileName.endswith(".tsv"):
                data = pd.read_csv(file, sep="\t")
                t_col, t_row = data.shape

                columnsNames = data.columns[1:]
                colNamesIndex = np.arange(1, t_row)

                data = data.to_numpy()

                new_max = np.amax(data[:, 1:], axis=1)

                coll = np.array([new_max == data[:, i] for i in range(1, t_row)]).T
                new_col = np.array([colNamesIndex[i][0] for i in coll])
                max_is_in_col[new_max > max_c] = new_col[new_max > max_c]

                n_com[new_max > max_c] = nComp
                max_c[new_max > max_c] = new_max[new_max > max_c]



    result_Ratio = np.c_[IDs, max_c, max_is_in_col, n_com]
    result_Ratio = pd.DataFrame(result_Ratio, columns=["Phenotype", "Max_corr", "NMF_id", "nComp"])
    for i in range(len(result_Ratio['NMF_id'])):
        result_Ratio['NMF_id'][i] = "NMF_"+str(int(result_Ratio['NMF_id'][i])).zfill(4)
    result_Ratio = result_Ratio.astype({"Phenotype": str, "Max_corr": float, "NMF_id": str,"nComp": int})

    print("\n----> Ratio\n", result_Ratio)


    if not os.path.exists("Calculations/NMF_TumorCell/Top_Correlations"):
        os.makedirs("Calculations/NMF_TumorCell/Top_Correlations")

    result_mean.to_csv("Calculations/NMF_TumorCell/Top_Correlations/NMF_Mean_Top_Correlations_spearman.tsv", sep='\t', index=False)
    result_Ratio.to_csv("Calculations/NMF_TumorCell/Top_Correlations/NMF_Ratio_Top_Correlations_spearman.tsv", sep='\t', index=False)

if __name__ == '__main__':
    main()