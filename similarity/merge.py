import numpy as np
from data import load_similarity_data


def merge_similarity(directory):
    disease_IDSSIM, disease_Gaussian, lncRNA_IDSSIM, lncRNA_Gaussian = load_similarity_data(directory)

    # Integrating Disease Similarity
    disease_IDSSIM = disease_IDSSIM.numpy()
    disease_Gaussian = disease_Gaussian.numpy()
    inter_disease = np.zeros(shape=(disease_IDSSIM.shape[0], disease_IDSSIM.shape[1]))

    for i in range(disease_IDSSIM.shape[0]):
        for j in range(disease_IDSSIM.shape[1]):
            if disease_IDSSIM[i][j] == 0:
                inter_disease[i][j] = disease_Gaussian[i][j]
            else:
                inter_disease[i][j] = (disease_IDSSIM[i][j] + disease_Gaussian[i][j]) / 2

    np.savetxt(directory + 'inter_disease'
               , inter_disease
               , delimiter=',', fmt='%.16f'
               )
    print("Merge Disease Finished!!!!")

    # Integrating Disease Similarity
    lncRNA_IDSSIM = lncRNA_IDSSIM.numpy()
    lncRNA_Gaussian = lncRNA_Gaussian.numpy()
    inter_lncRNA = np.zeros(shape=(lncRNA_IDSSIM.shape[0], lncRNA_IDSSIM.shape[1]))
    for i in range(lncRNA_IDSSIM.shape[0]):
        for j in range(lncRNA_IDSSIM.shape[1]):
            if lncRNA_IDSSIM[i][j] == 0:
                inter_lncRNA[i][j] = lncRNA_Gaussian[i][j]
            else:
                inter_lncRNA[i][j] = (lncRNA_IDSSIM[i][j] + lncRNA_Gaussian[i][j]) / 2

    np.savetxt(directory + 'inter_lncRNA'
               , inter_lncRNA
               , delimiter=',', fmt='%.16f'
               )
    print("Merge LncRNA Finished!!!!")


if __name__ == '__main__':
    merge_similarity('../data1/')
    print("Merge Similarity Finished!!!!")
