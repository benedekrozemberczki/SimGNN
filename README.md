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

A reference Tensorflow implementation is accessible [[here]](https://github.com/yunshengb/SimGNN).

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
### Datasets

The code takes pairs of graphs for training from an input folder where each pair of graph is stored as a JSON. Pairs of graphs used for testing are also stored as JSON files. Every node id and node label has to be indexed from 0. Keys of dictionaries and are stored strings in order to make JSON serialization possible.

Every JSON file has the following key-value structure:

```javascript
{"target": 1,
 "edges": [[0, 1], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
 "labels": {"0": 2, "1": 3, "2": 2, "3": 3, "4": 4},
 "inverse_labels": {"2": [0, 2], "3": [1, 3], "4": [4]}}
```
The **target key** has an integer value, which is the ID of the target class (e.g. Carcinogenicity). The **edges key** has an edge list value for the graph of interest. The **labels key** has a dictonary value for each node, these labels are stored as key-value pairs (e.g. node - atom pair). The **inverse_labels key** has a key for each node label and the values are lists containing the nodes that have a specific node label.

### Options

Training a SimGNN model is handled by the `src/main.py` script which provides the following command line arguments.

#### Input and output options

```
  --train-graph-folder   STR    Training graphs folder.      Default is `input/train/`.
  --test-graph-folder    STR    Testing graphs folder.       Default is `input/test/`.
```

#### Model options

```
  --repetitions          INT         Number of scoring runs.                  Default is 10. 
  --batch-size           INT         Number of graphs processed per batch.    Default is 32. 
  --time                 INT         Time budget.                             Default is 20. 
  --step-dimensions      INT         Neurons in step layer.                   Default is 32. 
  --combined-dimensions  INT         Neurons in shared layer.                 Default is 64. 
  --epochs               INT         Number of GAM training epochs.           Default is 10. 
  --learning-rate        FLOAT       Learning rate.                           Default is 0.001.
  --gamma                FLOAT       Discount rate.                           Default is 0.99. 
  --weight-decay         FLOAT       Weight decay.                            Default is 10^-5. 
```

### Examples
The following commands learn a neural network and score on the test set.

Training a SimGNN model on the default dataset. Saving predictions and logs at default paths.
```
python src/main.py
```
<p align="center">
<img style="float: center;" src="simgnn_run.jpg">
</p>

Training a GAM model for a 100 epochs with a batch size of 512.
```
python src/main.py --epochs 100 --batch-size 512
```
Setting a high time budget for the agent.
```
python src/main.py --time 128
```
Training a model with some custom learning rate and epoch number.
```
python src/main.py --learning-rate 0.001 --epochs 200
```
