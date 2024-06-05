import pandas as pd
import gaussian


def similarity(directory):

    association_matrix = pd.read_excel(directory + 'lncRNADisease-lncRNA-disease associations matrix.xls'
                                       , header=0, index_col=0)

    result_lncRNA = gaussian.gaussian_similarity(association_matrix, 'lncRNA')
    result_lncRNA.to_csv(directory + 'Gaussian_lncRNA_similarity.csv'
                         , index=False
                         , header=False
                         , float_format='%.16f')

    result_disease = gaussian.gaussian_similarity(association_matrix, 'disease')
    result_disease.to_csv(directory + 'Gaussian_disease_similarity.csv'
                          , index=False
                          , header=False
                          , float_format='%.16f')


if __name__ == '__main__':
    directory = '../../data1/'
    similarity(directory)
    print('Gaussian Similarity Finished!!!!')
