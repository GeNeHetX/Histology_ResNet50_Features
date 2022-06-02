import pandas as pd
import os
from sklearn.decomposition._nmf import _beta_divergence
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from joblib import parallel_backend


def main():
    train_sum_of_errors=[]
    test_sum_of_errors=[]

    print('-> Reading file...\n')
    # the number of row in each data frame
    # you can put any value here according to your situation
    chunksize = 200000

    # the list that contains all the dataframes
    list_of_dataframes = []
    list_of_dataframes = []

    print('--File too large...Reading in chuncks\n')
    c=0
    for mini_df in pd.read_csv('WSI_Features/Total_TumorCell_Features_avg.tsv', sep='\t', engine='python', chunksize=chunksize):
        # process your data frame here
        # then add the current data frame into the list
        c=c+1
        print("chunk number: ", c)
        list_of_dataframes.append(mini_df)

    # if you want all the dataframes together, here it is
    print('##Merging DataFrame##')
    df = pd.concat(list_of_dataframes)
    patch_id = df.iloc[:, :3]
    patch_id.reset_index(drop=True, inplace=True)

    #sample=int(input("Input sample: "))
    sample=200000
    data_train = df.sample(sample)
    data_train=data_train.drop(['WSI_id','X','Y','Tissue_score'], axis=1)


    n_components_List=list(range(2,201))
    n_components_df = pd.DataFrame(n_components_List, columns = ['N_Comp' ])


    for n_components in n_components_List:
        print("---> n_components = ", n_components)
    
        name_list=[]
        padding=4
        for i in range(1, n_components+1):
            name="NMF_"+str(i).zfill(padding)
            name_list.append(name)

        print("\n-> Transforming...\n")

        with parallel_backend('threading', n_jobs=16):
            # Your scikit-learn code here
            model = NMF(n_components=n_components, max_iter=1000, verbose=1, solver='mu')
            X = model.fit(data_train)

        list_of_W = []
        for data_test in [df[i:i+chunksize] for i in range(0,df.shape[0],chunksize)]:
            W = X.transform(data_test.iloc[:, 4:])
            W = pd.DataFrame(data= W, columns= name_list)
            list_of_W.append(W)

        All_W = pd.concat(list_of_W)
        All_W.reset_index(drop=True, inplace=True)

        H = model.components_

        print("\n-> Saving file\n")
        H = pd.DataFrame(H)
        All_W = pd.concat([patch_id, All_W], axis=1)

        output_subfolder= 'Calculations/NMF_TumorCell/nComp'+str(n_components)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        H.to_csv(output_subfolder+'/NMF_H_ALL_nComp'+str(n_components)+'.tsv', sep='\t', index=False)
        All_W.to_csv(output_subfolder+'/NMF_W_ALL_nComp'+str(n_components)+'.tsv', sep='\t', index=False)


        train_rec_error = X.reconstruction_err_
        print('Manually calculated rec-error train: ', train_rec_error)
        train_sum_of_errors.append(train_rec_error)

        test_rec_error = _beta_divergence(df.iloc[:, 4:], All_W.iloc[:, 3:], X.components_, 'frobenius', square_root=True)
        print('Manually calculated rec-error test: ', test_rec_error)
        test_sum_of_errors.append(test_rec_error)


    train_sum_of_errors = pd.DataFrame(train_sum_of_errors, columns = ["train_rec_error"])
    test_sum_of_errors = pd.DataFrame(test_sum_of_errors, columns = ["test_rec_error"])
    df2 = pd.DataFrame(pd.concat([n_components_df,train_sum_of_errors,test_sum_of_errors], axis=1))
    df2.to_csv('Calculations/NMF_TumorCell/NMF_reconstruction_errors.tsv', sep='\t', index=False)

if __name__ == '__main__':
    main()