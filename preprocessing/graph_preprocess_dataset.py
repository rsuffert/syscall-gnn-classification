import argparse
import os
import pickle

from preprocessing import GraphEncoder, SyscallFileReader


def process_subfolder(reader, encoder, subfolder_path, filter_calls, plot_graphs, custom_label=None):
    subfolder_graph_data = []

    for filename in os.listdir(subfolder_path):
        file_path = os.path.join(subfolder_path, filename)

        print(file_path)
        syscalls = reader.read(file_path)
        graph, node_mapping = encoder.encode(syscalls)

        if plot_graphs:
            label = custom_label if custom_label else os.path.basename(subfolder_path)
            encoder.plot_graph(file_path, filter_calls, graph, node_mapping)

        # Append graph and label to the list
        label = custom_label if custom_label else os.path.basename(subfolder_path)
        subfolder_graph_data.append({'graph': graph, 'label': label})

    return subfolder_graph_data


def preprocess_dataset(dataset_folder, filter_calls, plot_graphs, output_filename, vocab=None):
    all_graph_data = []
    reader = SyscallFileReader(filter_calls)
    encoder = GraphEncoder(vocab)

    # Check if processing ADFA-LD dataset
    if "ADFA" in dataset_folder:
        main_dirs = ['Attack_Data_Master', 'Training_Data_Master', 'Validation_Data_Master']
        for subfolder_name in os.listdir(dataset_folder):
            subfolder_path = os.path.join(dataset_folder, subfolder_name)
            if os.path.isdir(subfolder_path) and subfolder_name in main_dirs:
                if subfolder_name == 'Attack_Data_Master':
                    # Process each attack type subfolder within Attack_Data_Master
                    for attack_type in os.listdir(subfolder_path):
                        attack_subfolder_path = os.path.join(subfolder_path, attack_type)
                        label = f'attack_{attack_type}'
                        all_graph_data.extend(process_subfolder(reader, encoder, attack_subfolder_path, filter_calls, plot_graphs, custom_label=label))
                else:
                    # Directly process files within Training_Data_Master and Validation_Data_Master, labeling them as 'normal'
                    label = 'normal'
                    all_graph_data.extend(process_subfolder(reader, encoder, subfolder_path, filter_calls, plot_graphs, custom_label=label))
    else:
        # Process dummy dataset structure
        for subfolder_name in os.listdir(dataset_folder):
            subfolder_path = os.path.join(dataset_folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                all_graph_data.extend(process_subfolder(reader, encoder, subfolder_path, filter_calls, plot_graphs))

    # Serialize all graph data to a single file at the specified location
    output_file_path = os.path.join(dataset_folder, output_filename)
    with open(output_file_path, 'wb') as f:
        # Create a dictionary to hold both the graph data and additional information
        data_to_save = {
            'graph_data': all_graph_data,
            'vocab_size': len(encoder.syscall_vocab),
            'vocab': encoder.syscall_vocab
        }
        pickle.dump(data_to_save, f)

    return output_file_path


def load_and_print_dataset(file_path):
    # Load the dataset from the specified pickle file
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    graph_data = loaded_data['graph_data']
    vocab_size = loaded_data['vocab_size']

    # Print the contents of the loaded dataset
    for item in graph_data:
        print(f"Graph: {item['graph']}\nLabel: {item['label']}\n")

    print(vocab_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a dataset of system call logs and generate graph representations.')
    parser.add_argument('dataset_folder', type=str, help='Path to the dataset folder')
    parser.add_argument('--filter', action='store_true', help='Filter to include only relevant system calls')
    parser.add_argument('--plot', action='store_true', help='Generate and save plots of the syscall graphs')
    parser.add_argument('--output', type=str, default='processed_graphs.pkl', help='Output file for processed graphs')

    args = parser.parse_args()

    output_file_path = preprocess_dataset(args.dataset_folder, args.filter, args.plot, args.output)
    load_and_print_dataset(output_file_path)
