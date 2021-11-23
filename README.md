# GNN Artwork Classification
The aim of this project is to develop an artwork classifier (a graph neural network) that uses both visual features and contextual information in form of KG (knowledge graph). Given an artwork image, the classifier want to predict the artist, the style and the genre. The KG used is ArtGraph (https://arxiv.org/abs/2105.15028). 

## Artgraph
ArtGraph is a knowledge graph based on WikiArt and DBpedia, that integrates a rich body of information about artworks, artists, painting schools, etc.
In recent works ArtGraph has been used for multi-output classification tasks. The objective is that given an artwork we want to know its artist, style and genre.

## Requirements

pytorch 1.9

pytorch_geometric 2.0.1

mlflow 1.2.0

## Usage
Clone the repository, then

`cd gcnboost/src`

### Run a single experiment: 

```
python main.py 
    --exp <experiment_name> #will be used by mlflow
    --type <graph_type> #(hetero|homo)
    --mode <mode> # (single_task|multi_task)
    --label <label_name> #if single_task (artist|style|genre); ignore if multi_task
    --epochs <number_of_epochs> 
    --lr <learning_rate> 
    --hidden <number_of_hidden_units> #of hidden layers
    --nlayers <number_of_layers> #excluding input and output layers
    --dropout <dropout_probability>
    --operator <convolutional_operator> #one among (SAGEConv|GraphConv|GATConv|GCNConv);
     note that GCNConv can't be used for heterogenous graphs
    --aggr <aggregation_function> #one between (mean|sum)
    --skip #if you want to add skip connections
```

### Run a batch of experiments:

You can run a batch of experiments by executing the run_*.sh scripts (you can edit them and put the parameters you want). In particular:

* `run_hetero_multi-task.sh`, run experiments on a heterogeneous graph on a multilabel classification task.

* `run_hetero_single-task.sh`, run experiments on a heterogeneous graph on a single label classification task.

* `run_homo_multi-task.sh`, run experiments on a homogenous graph (ArtGraph is converted into a homogeneous graph, since it is originally heterogeneous) on a multilabel classification task.

* `run_homo_single-task.sh`, run experiments on a homogeneous graph on a single label classification task.

## GCNBoost
This work try to reproduce the GCNBoost system (https://arxiv.org/abs/2105.11852).

## Authors and acknowledgment
Vincenzo Digeno, v.digeno@studenti.uniba.it

Prof. Gennaro Vessio