import mxnet as mx
import warnings

from train import Train


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    Train(directory='../data1/',
          epochs=200,
          aggregator='GraphSAGE',  # 'GraphSAGE'
          embedding_size=128,
          layers=2,
          dropout=0.7,
          slope=0.2,  # LeakyReLU
          lr=0.001,
          wd=1e-3,
          random_seed=1234,
          ctx=mx.gpu(0))


