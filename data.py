import pandas as pd


# Data Reading
def load_data(directory):
    lncRNA_feature = pd.read_csv(directory + 'r128_feature.csv', header=None, index_col=None).to_numpy()
    disease_feature = pd.read_csv(directory + 'd128_feature.csv', header=None, index_col=None).to_numpy()
    return lncRNA_feature, disease_feature


# Selection of positive and negative samples
def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + 'pair.txt', sep=' ', names=['lncRNA', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)
    sample_df = known_associations._append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    return sample_df.values
