import numpy as np
import pandas as pd
import mxnet as mx
from mxnet import ndarray as nd
import dgl
from data import load_similarity_data


def load_data(directory):
    disease_IDSSIM, disease_Gaussian, lncRNA_IDSSIM, lncRNA_Gaussian = load_similarity_data(directory)

    ID = np.zeros(shape=(disease_IDSSIM.shape[0], disease_IDSSIM.shape[1]))
    for i in range(disease_IDSSIM.shape[0]):
        for j in range(disease_IDSSIM.shape[1]):
            if disease_IDSSIM[i][j] == 0:
                ID[i][j] = disease_Gaussian[i][j]
            else:
                ID[i][j] = (disease_IDSSIM[i][j] + disease_Gaussian[i][j]) / 2

    IL = np.zeros(shape=(lncRNA_IDSSIM.shape[0], lncRNA_IDSSIM.shape[1]))
    for i in range(lncRNA_IDSSIM.shape[0]):
        for j in range(lncRNA_IDSSIM.shape[1]):
            if lncRNA_IDSSIM[i][j] == 0:
                IL[i][j] = lncRNA_Gaussian[i][j]
            else:
                IL[i][j] = (lncRNA_IDSSIM[i][j] + lncRNA_Gaussian[i][j]) / 2
    return ID, IL


def sample(directory, random_seed):
    all_associations = pd.read_csv(directory + 'pair.txt', sep=' ', names=['lncRNA', 'disease', 'label'])
    known_associations = all_associations.loc[all_associations['label'] == 1]
    unknown_associations = all_associations.loc[all_associations['label'] == 0]
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed, axis=0)
    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    return sample_df.values


def build_graph(directory, random_seed, ctx):
    # dgl.load_backend('mxnet')
    ID, IL = load_data(directory)
    samples = sample(directory, random_seed)

    print('Building graph ...')
    g = dgl.DGLGraph(multigraph=True)
    g.add_nodes(ID.shape[0] + IL.shape[0])
    node_type = nd.zeros(g.number_of_nodes(), dtype='float32', ctx=ctx)
    node_type[:ID.shape[0]] = 1
    g = g.to(ctx)
    g.ndata['type'] = node_type

    print('Adding disease features ...')
    d_data = nd.zeros(shape=(g.number_of_nodes(), ID.shape[1]), dtype='float32', ctx=ctx)
    d_data[: ID.shape[0], :] = nd.from_numpy(ID)
    g.ndata['d_features'] = d_data

    print('Adding lncRNA features ...')
    m_data = nd.zeros(shape=(g.number_of_nodes(), IL.shape[1]), dtype='float32', ctx=ctx)
    m_data[ID.shape[0]: ID.shape[0] + IL.shape[0], :] = nd.from_numpy(IL)
    g.ndata['l_features'] = m_data

    print('Adding edges ...')
    disease_ids = list(range(1, ID.shape[0] + 1))
    lncrna_ids = list(range(1, IL.shape[0] + 1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    lncrna_ids_invmap = {id_: i for i, id_ in enumerate(lncrna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_lncrna_vertices = [lncrna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_lncrna_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.add_edges(sample_lncrna_vertices, sample_disease_vertices,
                data={'inv': nd.zeros(samples.shape[0], dtype='int32', ctx=ctx),
                      'rating': nd.from_numpy(samples[:, 2].astype('float32')).copyto(ctx)})
    g.readonly()
    print('Successfully build graph !!')

    return g
