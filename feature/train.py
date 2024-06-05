import pickle
import time
import random
import mxnet.model
import numpy as np
import pandas as pd
import math
import mxnet as mx
from mxnet import ndarray as nd, gluon, autograd
from mxnet.gluon import loss as gloss
import dgl
from utils import build_graph
from model import GNNMDA, GraphEncoder, BilinearDecoder


def Train(directory, epochs, aggregator, embedding_size, layers, dropout, slope, lr, wd, random_seed, ctx):
    dgl.load_backend('mxnet')
    random.seed(random_seed)
    np.random.seed(random_seed)
    mx.random.seed(random_seed)

    g = build_graph(directory, random_seed=random_seed, ctx=ctx)
    print('## vertices:', g.number_of_nodes())
    print('## edges:', g.number_of_edges())
    print('## disease nodes:', nd.sum(g.ndata['type'] == 1).asnumpy())
    print('## lncrna nodes:', nd.sum(g.ndata['type'] == 0).asnumpy())

    label_edata = g.edata['rating']
    src, dst = g.all_edges()

    # Train the model
    model = GNNMDA(GraphEncoder(embedding_size=embedding_size, n_layers=layers, G=g, aggregator=aggregator,
                                dropout=dropout, slope=slope, ctx=ctx),
                   BilinearDecoder(feature_size=embedding_size))  # 指定编码器和解码器

    model.collect_params().initialize(init=mx.init.Xavier(magnitude=math.sqrt(2.0)), ctx=ctx)
    cross_entropy = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
    for epoch in range(epochs):
        start = time.time()
        for _ in range(10):
            with mx.autograd.record():
                score_train = model(g, src, dst)
                loss_train = cross_entropy(score_train, label_edata).mean()
                loss_train.backward()
            trainer.step(1)
        end = time.time()
        print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.asscalar(), 'Time: %.2f' % (end - start))
    print('## Training Finished !')

    # Feature Extraction
    h = model.encoder(g)
    arr = h.asnumpy()
    arr_disease = arr[:157, :]
    arr_lncRNA = arr[157:, :]
    print(f'disease.shape: {arr_disease.shape}')
    print(f'lncRNA.shape: {arr_lncRNA.shape}')
    np.savetxt(directory + 'd16_feature.csv', arr_disease, delimiter=',')
    np.savetxt(directory + 'r16_feature.csv', arr_lncRNA, delimiter=',')
    print('Feature Extraction Finished')

