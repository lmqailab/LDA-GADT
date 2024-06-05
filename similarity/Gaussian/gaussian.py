import numpy as np
import pandas as pd
import math


def gaussian_similarity(association_matrix, flag):
    temp = association_matrix.values
    width_count = temp.shape[1]
    temp = temp.T
    width = 0
    for c in range(width_count):
        width += np.sum(temp[c] ** 2) ** 0.5

    if flag == 'lncRNA':
        index = association_matrix.index.to_list()
        association_matrix = association_matrix.values
        n = association_matrix.shape[0]

    elif flag == 'disease':
        index = association_matrix.columns.to_list()
        association_matrix = association_matrix.values
        n = association_matrix.shape[1]
        association_matrix = association_matrix.T
    else:
        print("input error")
        return

    similarity_matrix = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            similarity_matrix[i, j] = math.exp((np.sum((association_matrix[i] - association_matrix[j]) ** 2) ** 0.5
                                                * width / n) * (-1))
    result = pd.DataFrame(similarity_matrix)
    return result
