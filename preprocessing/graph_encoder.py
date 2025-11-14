import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

FILE_SYSCALLS = {
    "open", "mkdir", "rmdir", "access", "chmod", "rename", "unlink", "getdents64", "dup", "pread", "pwrit", "fcntl64"
}

NETWORK_SYSCALLS = {
    "recvfrom", "write", "read", "sendto", "writev", "close", "socket", "bind", "connect",
    "recvmsg", "sendmsg", "epoll_wait"
}


def get_syscall_type_encoding(syscall):
    if isinstance(syscall, int):
        return []
    elif syscall in FILE_SYSCALLS:
        return [1, 0, 0]
    elif syscall in NETWORK_SYSCALLS:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


class GraphEncoder:
    UNKNOWN_KEY = "<UNK>"

    def __init__(self, syscall_vocab=None):
        self.syscall_vocab = syscall_vocab
        if not syscall_vocab:
            self.syscall_vocab = {}
        self.user_provided_vocab = len(self.syscall_vocab) != 0
        if self.UNKNOWN_KEY not in self.syscall_vocab:
            self.syscall_vocab[self.UNKNOWN_KEY] = len(self.syscall_vocab)

    def build_vocabulary(self, syscalls):
        # Build vocabulary from the list of syscalls
        for syscall in syscalls:
            if syscall not in self.syscall_vocab:
                # Assign a new token to the syscall
                self.syscall_vocab[syscall] = len(self.syscall_vocab)

    def encode(self, syscalls):
        if not self.user_provided_vocab:
            # Ensure the vocabulary is built
            self.build_vocabulary(syscalls)

        unique_syscalls = list(set(syscalls))
        num_nodes = len(unique_syscalls)
        node_mapping = {syscall: i for i, syscall in enumerate(unique_syscalls)}

        G = nx.DiGraph()
        edge_counter = {}
        for i in range(len(syscalls) - 1):
            src = node_mapping[syscalls[i]]
            dst = node_mapping[syscalls[i + 1]]
            edge = (src, dst)
            edge_counter[edge] = edge_counter.get(edge, 0) + 1
            if G.has_edge(src, dst):
                G[src][dst]['weight'] += 1
            else:
                G.add_edge(src, dst, weight=1)

        # Define nodes and their features
        node_features = []
        if num_nodes > 1:
            # Compute centrality measures and other features
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G, weight='weight')

            try:
                katz_centrality = nx.katz_centrality_numpy(G, weight='weight')
            except np.linalg.LinAlgError as e:
                print(f"Numpy Katz centrality computation failed for graph with nodes: {G.nodes()} and edges: {G.edges()}. Exception: {e}")
                print("Falling back to iterative Katz centrality computation...")
                try:
                    katz_centrality = nx.katz_centrality(G,alpha=0.001 ,weight='weight', max_iter=5000)
                except Exception as e:
                    print(f"Iterative Katz centrality computation also failed. Exception: {e}")
                    print("Falling back to 0 for all nodes.")
                    katz_centrality = {node: 0 for node in G}

            for syscall in unique_syscalls:
                node_idx = node_mapping[syscall]
                token = self.syscall_vocab.get(syscall, self.syscall_vocab[self.UNKNOWN_KEY])
                katz = katz_centrality[node_idx]
                betweenness = betweenness_centrality[node_idx]
                closeness = closeness_centrality[node_idx]
                pr = pagerank[node_idx]
                syscall_type_encoding = get_syscall_type_encoding(syscall)

                # Append features to the node features, including the syscall token
                features = [token] + syscall_type_encoding + [katz, betweenness, closeness, pr]
                node_features.append(features)
        else:
            # Handle singleton graph, assign default centrality measures
            for syscall in unique_syscalls:
                token = self.syscall_vocab.get(syscall, self.syscall_vocab[self.UNKNOWN_KEY])
                syscall_type_encoding = get_syscall_type_encoding(syscall)
                features = [token] + syscall_type_encoding + [0, 0, 0, 0]
                node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = list(G.edges())
        edge_features = [edge_counter[edge] for edge in edge_index]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_features = torch.tensor(edge_features, dtype=torch.float)

        graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes, edge_features=edge_features)
        graph.validate(raise_on_error=True)

        return graph, node_mapping

    def plot_graph(self, file_path, filter_calls, graph, node_mapping):
        g = torch_geometric.utils.to_networkx(graph, to_undirected=False)
        pos = nx.spring_layout(g)
        reverse_mapping = {i: syscall for syscall, i in node_mapping.items()}

        # Extract node colors based on their features
        # Compute node colors based on the one-hot encoding in positions [1, 3]
        node_colors = []
        for i in range(graph.num_nodes):
            encoding = graph.x[i][1:4].tolist()  # Extract the one-hot encoding
            if encoding == [1.0, 0.0, 0.0]:
                color = 0
            elif encoding == [0.0, 1.0, 0.0]:
                color = 1
            else:
                color = 2
            node_colors.append(color)

        # Extract edge labels based on the edge_features
        edge_labels = {tuple(graph.edge_index[:, i].tolist()): int(graph.edge_features[i].item()) for i in range(graph.edge_features.size(0))}

        # Define color map for nodes
        color_map = {0: '#3366CC', 1: '#DC3912', 2: '#FF9900'}

        # Set the figure size
        plt.figure(figsize=(10, 6))  # Adjust the size as needed

        # Draw nodes with different colors and labels
        nx.draw_networkx_nodes(g, pos, node_color=[color_map[color] for color in node_colors], node_size=100)
        labels = {node: reverse_mapping[node] for node in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)

        # Draw edges with labels
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)

        # Set the legend for node colors to the top-right
        node_labels = {'File syscall node': '#3366CC', 'Network syscall node': '#DC3912', 'Default (other) syscall node': '#FF9900'}
        patches = [plt.Line2D([], [], marker='o', color=color, markersize=8, linestyle='None', label=label) for label, color in node_labels.items()]
        plt.legend(handles=patches, title='Node Features', loc='upper right')

        plt.axis('off')
        plt.savefig(f'{file_path}.png') if not filter_calls else plt.savefig(f'{file_path}_filtered.png')


def test(file_path, filter_calls, plot):
    reader = SyscallFileReader(filter_calls)
    encoder = GraphEncoder()

    # Read syscall
    syscalls = reader.read(file_path)
    graph, node_mapping = encoder.encode(syscalls=syscalls)

    if plot:
        # Generate graph plot
        encoder.plot_graph(file_path, filter_calls, graph, node_mapping)


if __name__ == "__main__":
    # python main.py ./strace_generator/strace_output_example1 --filter

    parser = argparse.ArgumentParser(description='Process system call data and visualize as a graph.')
    parser.add_argument('file_path', type=str, help='Path to the file containing system calls')
    parser.add_argument('--filter', action='store_true', help='Filter to include only relevant system calls')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot the graph')

    args = parser.parse_args()

    test(args.file_path, args.filter, args.plot)
