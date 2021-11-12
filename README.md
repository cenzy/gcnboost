# GNN Artwork Classification
The aim of this project is to develop an artwork classifier (a graph neural network) that uses both visual features and contextual information in form of KG (knowledge graph). The KG used is ArtGraph.

## Artgraph
ArtGraph is a knowledge graph based on WikiArt and DBpedia, that integrates a rich body of information about artworks, artists, painting schools, etc.
In recent works ArtGraph has been used for multi-output classification tasks. The objective is that given an artwork we want to know its artist, style and genre.

## Usage
For a quick tour see this notebook https://colab.research.google.com/drive/1Fs1GWLmu_MbAenUthTUVTdjqYBI22boo?usp=sharing
### Import the packages
This project uses the frameworks PyTorch and PyG (Pytorch Geometric). PyG (PyTorch Geometric) is a library built upon PyTorch to easily write and train Graph Neural Networks (GNNs) for a wide range of applications related to structured data.

For more details on PyG installation refers to https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

``` 
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu111.html
!pip install -q git+https://github.com/rusty1s/pytorch_geometric.git

import torch
```
### Import the dataset
You can import the ArtGraph dataset as an instance of the torch_geometric.data.HeteroData, a data object type defined by PyG framework. For this type of object PyG provides a set of tools for managing the graph and implementing Graph Neural Newtwork architectures.

To each artwork is assigned 128 dimensional feature vector obtained from a pre-trained ResNet50 on ImaneNet dataset. For the other nodes you can assign 128 dimensional feature vector obtained after running node2vec on the whole graph.

The dataset can be imported by instantiating the ArtGraph class. It asks three parameters:

- root (required): the path folder in which the dataset will be downloaded or, if you already have the dataset on your machine, the dataset path.
- preprocess (optional): set "node2vec" for provide a features to all the nodes (without considering the artwork node since it already have features). If set None, no feature will be download.
- transform (optional): a transformation to be applied to the dataset. See https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html for a list of possible transformations.
- features (optional, default is True): set True if you want the visual features for artworks. Set False otherwise.
- type (optional, default is 'kg'): set 'kg' if you want the full KG with all relations (also for validation and test artworks). Set 'ekg' if you want the extended knowledge graph (see https://arxiv.org/abs/2105.11852)

If you want, you can download manually the whole dataset here https://drive.google.com/drive/folders/1WIiosAdYJV3kWxocAjnclQJ4oL0v9iu3?usp=sharing
Extract the artgraph.zip e put its path into the url parameter.

```
from artgraph import ArtGraph

dataset = ArtGraph("./dataset", preprocess="node2vec")
kg = dataset = data[0]
```


## Authors and acknowledgment
Vincenzo Digeno, v.digeno@studenti.uniba.it

## Project status
In progress

