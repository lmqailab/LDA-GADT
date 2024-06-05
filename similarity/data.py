import torch
import pandas as pd


def load_similarity_data(directory):

    # disease IDSSIM
    disease_IDSSIM = pd.read_excel(directory + 'lncRNADisease-disease semantic similarity matrix.xls',
                                   index_col=0).values
    disease_IDSSIM = torch.from_numpy(disease_IDSSIM)

    # lncRNA IDSSIM
    lncRNA_IDSSIM = pd.read_excel(directory + 'lncRNADisease-lncRNA functional similarity matrix.xls',
                                  index_col=0).values
    lncRNA_IDSSIM = torch.from_numpy(lncRNA_IDSSIM)

    # disease Gaussian
    disease_Gaussian = pd.read_csv(directory + 'Gaussian_disease_similarity.csv', header=None).values
    disease_Gaussian = torch.from_numpy(disease_Gaussian)

    # lncRNA Gaussian
    lncRNA_Gaussian = pd.read_csv(directory + 'Gaussian_lncRNA_similarity.csv', header=None).values
    lncRNA_Gaussian = torch.from_numpy(lncRNA_Gaussian)

    return disease_IDSSIM, disease_Gaussian, lncRNA_IDSSIM, lncRNA_Gaussian


if __name__ == '__main__':
    load_similarity_data('../data1/')
    print('Load similarity Finished!!!!')
