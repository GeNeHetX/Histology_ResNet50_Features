import glob

import pandas as pd

from tqdm import tqdm

def main():
    print("\n->Initiating...")

    window_size=224*2

    images=sorted(glob.glob("WSI_Features/*/"))

    c=0
    for image in images:
        c=c+1
        print("----> Image: ", image.split('/')[1]," Number", c, ' Out of', len(images))

        df = pd.read_csv('pacpaint_results/'+image.split('/')[1]+'.svs/tile_scores.csv', sep=',')
        df = df.loc[(df['pred_tumor'] >= 0.6) & (df['pred_tumor_cell'] >= 0.6)]

        df2 = pd.read_csv(image+image.split('/')[1]+'_Features_ResNet50_avg.tsv', sep='\t')

        new_df = pd.DataFrame(columns=df2.columns)
        

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            X, Y = int(row['x'])*window_size, int(row['y'])*window_size
            new_line = df2.loc[(df2['X'] == X) & (df2['Y'] == Y)]
            
            try:
                new_df.loc[new_line.index[0]] = new_line.iloc[0]
            except:
                continue

        print("\n->Saving tsv files in corresponding subfolder in WSI_Features Folder\n")
        new_df.to_csv(image+image.split('/')[1]+'_TumorCell_Features_ResNet50_avg.tsv', sep='\t', index=False)



if __name__ == '__main__':
    main()

    