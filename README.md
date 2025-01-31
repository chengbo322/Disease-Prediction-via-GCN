# Reproducibility of Disease Prediction via Graph Neural Networks
A simple implementation of the paper "Disease Prediction via Graph Neural Networks".

## Dependencies
The code was tested in Windows 11 with below installations:

python >=3.6

pytorch

numpy

sklearn

## Citation to the original paper
Sun Z, Yin H, Chen H, et al. Disease prediction via graph neural networks[J]. IEEE Journal of Biomedical and
Health Informatics, 2020, 25(3): 818-826.

## Link to the original paper’s repo
Original Repo: https://github.com/zhchs/Disease-Prediction-via-GCN

## Data format in data/sample_data folder
```shell script
"filename.nodes.pkl"
# list of node: [node1(str), node2(str), node3(str), ...]

"filename.adj.pkl"
# adj list of nodes: 
# {node1(str): [neighbor1(str), neighbor2(str), ...], node2: []...}

"filename.rare.label.pkl"
# rare flag, indicating whether a node is a rare disease (value=1) 
# or contains a rare disease, NumPy array of shape (N * 1) 

"filename.label.pkl"
# NumPy array of shape (N * D), N is node number 
# and D is the number of diseases

"filename.map.pkl"
# mapping node to index, {node(str): node_id(int), ...}

"filename.train.pkl"
# list of nodes for training, [node_idx_1(int), node_idx_2(int), ....]

"filename.test.pkl"
# list of nodes for testing, [node_idx_1(int), node_idx_2(int), ....]
```

## Run The Proposed Model
```shell script
python run_multi.py
```
## Run The Baseline Models (DT, SVM, RF, KNN and SGD)
```shell script
python Baselines.py
```
## Main Reproducibility Result
![image](https://user-images.githubusercontent.com/70998318/167065204-a795698c-e37a-4890-933f-93e3330ecd65.png)
*Top-K diseases in each patient’s prediction result

