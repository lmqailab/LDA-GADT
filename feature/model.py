import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np

from layers import GraphSageLayer


class GNNMDA(nn.Block):
    def __init__(self, encoder, decoder):
        super(GNNMDA, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, G, diseases, lncrnas):
        h = self.encoder(G)
        h_diseases = h[diseases]
        h_lncrnas = h[lncrnas]
        return self.decoder(h_diseases, h_lncrnas)


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.lncrna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)

        self.layers = nn.Sequential()
        for i in range(n_layers):
            if aggregator == 'GraphSAGE':
                self.layers.add(
                    GraphSageLayer(embedding_size, G, self.disease_nodes, self.lncrna_nodes, dropout, slope, ctx))
            else:
                raise NotImplementedError

        self.disease_emb = DiseaseEmbedding(embedding_size, dropout)
        self.lncrna_emb = LncrnaEmbedding(embedding_size, dropout)

    def forward(self, G):
        # Generate embedding on disease nodes and lncrna nodes
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.apply_nodes(lambda nodes: {'h': self.disease_emb(nodes.data)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h': self.lncrna_emb(nodes.data)}, self.lncrna_nodes)

        for layer in self.layers:
            layer(G)

        return G.ndata['h']


class DiseaseEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(DiseaseEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_disease = seq

    def forward(self, ndata):
        extra_repr = self.proj_disease(ndata['d_features'])

        return extra_repr


class LncrnaEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(LncrnaEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size, use_bias=False))
            seq.add(nn.Dropout(dropout))
        self.proj_lncrna = seq

    def forward(self, ndata):
        extra_repr = self.proj_lncrna(ndata['l_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(feature_size, feature_size))

    def forward(self, h_diseases, h_lncrnas):
        results_mask = self.activation((nd.dot(h_diseases, self.W.data()) * h_lncrnas).sum(1))

        return results_mask
