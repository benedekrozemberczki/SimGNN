SimGNN
============================================
A PyTorch implementation of "SimGNN: A Neural Network Approach to Fast Graph Similarity Computation" (WSDM 2019). 
<p align="center">
  <img width="800" src="simgnn.jpg">
</p>
<p align="justify">
Graph similarity search is among the most important graph-based applications, e.g. finding the chemical compounds that are most similar to a query compound. Graph similarity/distance computation, such as Graph Edit Distance (GED) and Maximum Common Subgraph (MCS), is the core operation of graph similarity search and many other applications, but very costly to compute in practice. Inspired by the recent success of neural network approaches to several graph applications, such as node or graph classification, we propose a novel neural network based approach to address this classic yet challenging graph problem, aiming to alleviate the computational burden while preserving a good performance. The proposed approach, called SimGNN, combines two strategies. First, we design a learnable embedding function that maps every graph into an embedding vector, which provides a global summary of a graph. A novel attention mechanism is proposed to emphasize the important nodes with respect to a specific similarity metric. Second, we design a pairwise node comparison method to sup plement the graph-level embeddings with fine-grained node-level information. Our model achieves better generalization on unseen graphs, and in the worst case runs in quadratic time with respect to the number of nodes in two graphs. Taking GED computation as an example, experimental results on three real graph datasets demonstrate the effectiveness and efficiency of our approach. Specifically, our model achieves smaller error rate and great time reduction compared against a series of baselines, including several approximation algorithms on GED computation, and many existing graph neural network based models. Our study suggests SimGNN provides a new direction for future research on graph similarity computation and graph similarity search.</p>

This repository provides a PyTorch implementation of SimGNN as described in the paper:

> SimGNN: A Neural Network Approach to Fast Graph Similarity Computation.
> Yunsheng Bai, Hao Ding, Song Bian, Ting Chen, Yizhou Sun, Wei Wang.
> WSDM, 2019.
> [[Paper]](http://web.cs.ucla.edu/~yzsun/papers/2019_WSDM_SimGNN.pdf)

A reference Tensorflow implementation is accessible [[here]](https://github.com/Eilene/GWNN).

### Requirements
The codebase is implemented in Python 3.5.2. package versions used for development are just below.
```
networkx          1.11
tqdm              4.28.1
numpy             1.15.4
pandas            0.23.4
texttable         1.5.0
scipy             1.1.0
argparse          1.1.0
torch             0.4.1
torch-scatter     1.0.4
torch-sparse      0.2.2
torchvision       0.2.1
scikit-learn      0.20.0
```
