import pickle

import torch
from torch_geometric.data import Dataset
from torch_geometric.utils import dropout_adj, dropout_edge


def drop_feature_except_token(x, drop_prob):
    # Create a dropout mask for features other than the syscall token
    drop_mask = torch.bernoulli(torch.full((x.size(0), x.size(1) - 1), 1 - drop_prob, dtype=torch.float32, device=x.device))

    # Apply the dropout mask to features, leaving the syscall token intact
    x[:, 1:] *= drop_mask

    return x


class CustomGraphDataset(Dataset):
    def __init__(self, graph_data, classes, root=None, transform=None, pre_transform=None, drop_edge_rate=0.0, drop_feature_rate=0.0, training=True):
        super(CustomGraphDataset, self).__init__(root, transform, pre_transform)
        self.graph_data = graph_data
        self.classes = classes
        self.drop_edge_rate = drop_edge_rate
        self.drop_feature_rate = drop_feature_rate
        self.training = training

    @property
    def num_classes(self):
        return self.classes

    def len(self):
        return len(self.graph_data)

    def get(self, idx):
        data = self.graph_data[idx]

        # Check for NaNs and Infs in the data
        # assert not torch.isnan(data.x).any(), f"NaN found in node features of graph {idx}"
        # assert not torch.isinf(data.x).any(), f"Inf found in node features of graph {idx}"
        # assert not torch.isnan(data.y).any(), f"NaN found in labels of graph {idx}"
        # assert not torch.isinf(data.y).any(), f"Inf found in labels of graph {idx}"

        if self.training:
            # Apply edge dropout directly to the data's edge_index
            if self.drop_edge_rate > 0.0:
                data.edge_index, _ = dropout_edge(data.edge_index, p=self.drop_edge_rate, force_undirected=False, training=self.training)

            # Apply feature dropout directly to the data's features
            if self.drop_feature_rate > 0.0:
                data.x = drop_feature_except_token(data.x, self.drop_feature_rate)

        return data


def load_dataset(file_path):
    # Load the dataset and additional information from the specified pickle file
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    # Unpack the loaded data
    graph_data = loaded_data['graph_data']
    vocab_size = loaded_data['vocab_size']
    vocab      = loaded_data['vocab']

    # Process and return the unpacked data as needed
    return graph_data, vocab_size, vocab
