import numpy as np
import pandas as pd
import shutil
import torch
import torch_geometric as pyg
import os
from torch_geometric.data import (InMemoryDataset, HeteroData, download_url,
                                  extract_zip)

class ArtGraph(InMemoryDataset):

    url = {
        'kg': 'http://bicytour.altervista.org/artgraph/',
        'ekg': 'http://bicytour.altervista.org/artgraph-ekg/'
    }
    embedding_urls = {
        'node2vec': 'artgraph_node2vec_emb.zip'
    }

    def __init__(self, root, preprocess='node2vec', transform=None,
                 pre_transform=None, features=True, type='kg'):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        self.features = features
        self.type = type

        assert self.preprocess in [None, 'node2vec']
        assert type in ['kg', 'ekg']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'artgraph', 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv.gz'
        ]

        if self.preprocess is not None:
            file_names += ["artgraph_node2vec_emb.pt"]

        return file_names

    @property
    def processed_file_names(self):
        if self.preprocess is not None:
            return "artgraph_node2vec_emb.pt"
        else:
            return 'none.pt'
    
    def download(self):
        if not all([os.path.exists(f) for f in self.raw_paths[:5]]):
            path = download_url(os.path.join(self.url[self.type], 'artgraph.zip'), self.raw_dir)
            extract_zip(path, self.raw_dir)
            for file_name in ['node-feat', 'node-label', 'relations']:
                path = os.path.join(self.raw_dir, 'artgraph', 'raw', file_name)
                shutil.move(path, self.raw_dir)
            path = os.path.join(self.raw_dir, 'artgraph', 'split')
            shutil.move(path, self.raw_dir)
            path = os.path.join(self.raw_dir, 'artgraph', 'raw', 'num-node-dict.csv.gz')
            shutil.move(path, self.raw_dir)
            shutil.rmtree(os.path.join(self.raw_dir, 'artgraph'))
            os.remove(os.path.join(self.raw_dir, 'artgraph.zip'))
        if self.preprocess is not None:
            path = download_url(os.path.join(self.url[self.type], self.embedding_urls[self.preprocess]), self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self):
        data = pyg.data.HeteroData()
        if self.features == True:
            path = os.path.join(self.raw_dir, 'node-feat', 'artwork', 'node-feat.csv.gz')
            x_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values
            data['artwork'].x = torch.from_numpy(x_artwork)

        else:
            path = os.path.join(self.raw_dir, 'num-node-dict.csv.gz')
            num_nodes_df = pd.read_csv(path, compression="gzip")
            data['artwork'].num_nodes = num_nodes_df['artwork'].tolist()[0]

        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-artist.csv.gz')
        artist_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_artist = torch.from_numpy(artist_artwork)

        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-style.csv.gz')
        style_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_style = torch.from_numpy(style_artwork)

        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-genre.csv.gz')
        genre_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_genre = torch.from_numpy(genre_artwork)

        if self.preprocess is None:
            path = os.path.join(self.raw_dir, 'num-node-dict.csv.gz')
            num_nodes_df = pd.read_csv(path, compression="gzip")
            num_nodes_df.rename(columns={"training": "training_node"}, inplace=True)
            nodes_type = ['artist', 'gallery', 'city', 'country', 'style', 'period', 'genre', 'serie', 'auction', 'tag',
            	          'media', 'subject', 'training_node', 'field', 'movement', 'people']
            for node_type in nodes_type:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        else: #a relations is called 'training' and can create conflict with the concept of training a model hence we change its name
            emb_dict = torch.load(self.raw_paths[-1])
            for key, value in emb_dict.items():
                if key != 'artwork':
                    if key =='training':
                        key = 'training_node'
                    data[key].x = value   
		
        for edge_type in [('artist', 'influenced', 'artist'), 
                          ('artist', 'subject', 'subject'), 
                          ('artist', 'training', 'training'), 
                          ('artist', 'field', 'field'), 
                          ('artist', 'movement', 'movement'), 
                          ('artist', 'patrons', 'people'), 
                          ('artist', 'teacher', 'artist'), 
                          ('gallery', 'city', 'city'), 
                          ('city', 'country', 'country'), 
                          ('gallery', 'country', 'country'), 
                          ('artwork', 'media', 'media'), 
                          ('artwork', 'about', 'tag'), 
                          ('artwork', 'genre', 'genre'), 
                          ('artwork', 'style', 'style'), 
                          ('artwork', 'author', 'artist'), 
                          ('artwork', 'period', 'period'), 
                          ('artwork', 'locatedin', 'gallery'), 
                          ('artwork', 'auction', 'auction'), 
                          ('artwork', 'serie', 'serie'), 
                          ('artwork', 'completedin', 'city'), 
                          ('people', 'patrons', 'artist')]:
            f = '___'.join(edge_type)
            path = os.path.join(self.raw_dir, 'relations', f, 'edge.csv.gz')
            edge_index = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            h, r, t = edge_type
            if t == 'training':
                t = 'training_node'
            edge_type = (h, r + '_rel', t)
            data[edge_type].edge_index = edge_index
                
        for f, v in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            path = os.path.join(self.raw_dir, 'split', 'artwork',
                            f'{f}.csv.gz')
            idx = pd.read_csv(path, compression='gzip', header=None,
                              dtype=np.int64).values.flatten()
            idx = torch.from_numpy(idx)
            mask = torch.zeros(data['artwork'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['artwork'][f'{v}_mask'] = mask 
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self): 
        return {
           'artist': self.data['artist'].x.shape[0],
           'style': self.data['style'].x.shape[0],
           'genre': self.data['genre'].x.shape[0]
        }

    @property
    def num_features(self):
        return self.data['artist'].x.shape[1]

class ArtGraphV2():
    
    def __init__(self, root, preprocess="node2vec", transform=None,
                 pre_transform=None):
        preprocess = None if preprocess is None else preprocess.lower()
        self.preprocess = preprocess
        assert self.preprocess in [None, 'node2vec']
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    

    @property
    def raw_dir(self):
        return os.path.join(self.root, 'artgraph_2_0', 'raw')

    @property
    def raw_file_names(self):
        file_names = [
            'node-feat', 'node-label', 'relations', 'split',
            'num-node-dict.csv.gz'
        ]

        if self.preprocess is not None:
            file_names += ["artgraph_node2vec_emb.pt"]

        return file_names

    @property
    def processed_file_names(self):
        if self.preprocess is not None:
            return "artgraph_node2vec_emb.pt"
        else:
            return 'none.pt'
    
    def download(self):
        if not all([os.path.exists(f) for f in self.raw_paths[:5]]):
            path = download_url(self.url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            for file_name in ['node-feat', 'node-label', 'relations']:
                path = os.path.join(self.raw_dir, 'artgraph_2_0', 'raw', file_name)
                shutil.move(path, self.raw_dir)
            path = os.path.join(self.raw_dir, 'artgraph_2_0', 'split')
            shutil.move(path, self.raw_dir)
            path = os.path.join(self.raw_dir, 'artgraph_2_0', 'raw', 'num-node-dict.csv.gz')
            shutil.move(path, self.raw_dir)
            shutil.rmtree(os.path.join(self.raw_dir, 'artgraph_2_0'))
            os.remove(os.path.join(self.raw_dir, 'artgraph_2_0.zip'))
        if self.preprocess is not None:
            path = download_url(self.urls[self.preprocess], self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.remove(path)

    def process(self):
        data = pyg.data.HeteroData()
        
        path = os.path.join(self.raw_dir, 'node-feat', 'artwork', 'node-feat.csv.gz')
        x_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values
        data['artwork'].x = torch.from_numpy(x_artwork)
		
        path = os.path.join(self.raw_dir, 'node-label', 'artwork', 'node-label-emotion.csv.gz')
        emotion_artwork = pd.read_csv(path, compression="gzip", header=None, dtype=np.float32).values.flatten()
        data['artwork'].y_emotion = torch.from_numpy(emotion_artwork)

        if self.preprocess is None:
            path = os.path.join(self.raw_dir, 'num-node-dict.csv.gz')
            num_nodes_df = pd.read_csv(path, compression="gzip")
            num_nodes_df = num_nodes_df.rename(columns={"training": "training_node"})
            nodes_type = ['artist', 'gallery', 'city', 'country', 'style', 'period', 'genre', 'serie', 'auction', 'tag',
            	          'media', 'subject', 'training_node', 'field', 'movement', 'people', 'emotion']
            for node_type in nodes_type:
                data[node_type].num_nodes = num_nodes_df[node_type].tolist()[0]
        else:
            emb_dict = torch.load(self.raw_paths[-1])
            for key, value in emb_dict.items():
                if key != 'artwork':
                    if key =='training':
                        key = 'training_node'
                    data[key].x = value   
					
        for edge_type in [('artist', 'influenced', 'artist'), 
                          ('artist', 'subject', 'subject'), 
                          ('artist', 'training', 'training'), 
                          ('artist', 'field', 'field'), 
                          ('artist', 'movement', 'movement'), 
                          ('artist', 'patrons', 'people'), 
                          ('artist', 'teacher', 'artist'), 
                          ('gallery', 'city', 'city'), 
                          ('city', 'country', 'country'), 
                          ('gallery', 'country', 'country'), 
                          ('artwork', 'media', 'media'), 
                          ('artwork', 'about', 'tag'), 
                          ('artwork', 'genre', 'genre'), 
                          ('artwork', 'style', 'style'), 
                          ('artwork', 'author', 'artist'), 
                          ('artwork', 'period', 'period'), 
                          ('artwork', 'locatedin', 'gallery'), 
                          ('artwork', 'auction', 'auction'), 
                          ('artwork', 'serie', 'serie'), 
                          ('artwork', 'completedin', 'city'),
                          ('artwork', 'elicit', 'emotion'),
                          ('people', 'patrons', 'artist')]:
            f = '___'.join(edge_type)
            path = os.path.join(self.raw_dir, 'relations', f, 'edge.csv.gz')
            edge_index = pd.read_csv(path, compression='gzip', header=None, dtype=np.int64).values
            edge_index = torch.from_numpy(edge_index).t().contiguous()
            h, r, t = edge_type
            if t == 'training':
                t = 'training_node'
            edge_type = (h, r + '_rel', t)
            data[edge_type].edge_index = edge_index
            if r == 'elicit':
                edge_attr_path = os.path.join(self.raw_dir, 'relations', f, 'edge_attr.csv.gz')
                edge_attr = pd.read_csv(edge_attr_path, compression='gzip', header=None, dtype=np.int64).values
                edge_attr = torch.from_numpy(edge_attr).contiguous()
                edge_attr_type = (h, r + '_rel_attr', t)
                data[edge_attr_type].edge_attr = edge_attr
               
        for f, v in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
            path = os.path.join(self.raw_dir, 'split', 'artwork',
                            f'{f}.csv.gz')
            idx = pd.read_csv(path, compression='gzip', header=None,
                              dtype=np.int64).values.flatten()
            idx = torch.from_numpy(idx)
            mask = torch.zeros(data['artwork'].num_nodes, dtype=torch.bool)
            mask[idx] = True
            data['artwork'][f'{v}_mask'] = mask 
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)
        
        torch.save(self.collate([data]), self.processed_paths[0])

    @property
    def num_classes(self):
        return {
            'artist': self.data['artist'].x.shape[0],
            'style': self.data['style'].x.shape[0],
            'genre': self.data['genre'].x.shape[0],
            'emotion': self.data['emotion'].x.shape[0],
         }

    @property
    def num_features(self):
        return self.data['artist'].x.shape[1]




    
