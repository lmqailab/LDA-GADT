## Dataset

In this work，lncRNADisease is data1 and MNDR is data2.

## Environment

 - python $\approx$ 3.10
 - pytorch $\approx$ 2.3.0
 - tensorflow $\approx$ 2.13.0
 - numpy $\approx$ 1.24.3
 - pandas $\approx$ 2.2.2
 - scikit-learn $\approx$ 1.4.2

## Usage

Default is 5-fold cross validation on lncRNADisease  and MNDR databases. To run this model：
```
python main.py
```
Extracting features for diseases and lncRNAs by GAE, to run:
```
python feature/GAE.py
```
