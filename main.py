import os
import h5py
import logging
import argparse
import subprocess
from typing import Dict
from preprocessing.graph_preprocess_dataset import preprocess_dataset

NORMAL_TRAIN_H5 = os.getenv("NORMAL_TRAIN_H5", "Normal_DTDS-train.h5")
NORMAL_VALID_H5 = os.getenv("NORMAL_VALID_H5", "Normal_DTDS-validation.h5")
NORMAL_TEST_H5  = os.getenv("NORMAL_TEST_H5",  "Normal_DTDS-test.h5")
ATTACK_TRAIN_H5 = os.getenv("ATTACK_TRAIN_H5", "Attach_DTDS-train.h5")
ATTACK_VALID_H5 = os.getenv("ATTACK_VALID_H5", "Attach_DTDS-validation.h5")
ATTACK_TEST_H5  = os.getenv("ATTACK_TEST_H5",  "Attach_DTDS-test.h5")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end processing and training script for syscall graph classification")
    parser.add_argument("--extract", "-e", action="store_true", help="Run H5 to trace extraction")
    parser.add_argument("--preprocess", "-p", action="store_true", help="Run trace to graph preprocessing")
    parser.add_argument("--train", "-t", action="store_true", help="Run graph model training")
    return parser.parse_args()

def convert_h5_to_traces(h5_path: str, output_dir: str, syscall_tbl_path: str = "syscall_64.tbl"):
    def parse_syscall_tbl(path: str) -> Dict[int, str]:
        """
        Parses a system call table file and returns a mapping from syscall IDs to their names.

        Args:
            path (str): The path to the system call table file. Each line in the file should contain
                at least three fields:
                - The first field is the syscall ID (integer).
                - The third field is the syscall name (string).
                Lines starting with "#" or empty lines are ignored.

        Returns:
            Dict[int, int]: A dictionary mapping syscall IDs to their corresponding names.
        """
        syscalls_map = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                syscall_id = int(parts[0])
                syscall_name = parts[2]
                syscalls_map[syscall_id] = syscall_name
        return syscalls_map
    
    syscall_map = parse_syscall_tbl("syscall_64.tbl")

    os.makedirs(output_dir, exist_ok=True)
    with h5py.File(h5_path, "r") as h5f:
        sequences = h5f["sequences"]
        for idx, seq in enumerate(sequences):
            syscall_names = [syscall_map.get(int(sid), f"unknown_{sid}") for sid in seq]
            trace_path = os.path.join(output_dir, f"trace_{idx}.txt")
            with open(trace_path, "w") as f:
                for name in syscall_names:
                    f.write(f"{name}(\n") # one per line with '(' to match expected format

def preprocess_traces_to_graphs():
    dataset_folder = "traces"
    filter_calls = False
    plot_graphs = False
    output_filename = "processed_graphs.pkl"
    output_filepath = preprocess_dataset(dataset_folder, filter_calls, plot_graphs, output_filename)
    print(f"Graphs saved to {output_filepath}")

def train_gnn_model():
    subprocess.run([
        "python", "train_graph.py",
        "--dataset_path", "traces/processed_graphs.pkl",
        "--model", "GIN",
        "--epochs", "250",
        "--batch_size", "256"
    ], check=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    args = parse_args()
    if args.extract:
        logging.info("Converting H5 files to trace files...")
        convert_h5_to_traces(NORMAL_TRAIN_H5, "traces/normal/train")
        convert_h5_to_traces(NORMAL_VALID_H5, "traces/normal/validation")
        convert_h5_to_traces(NORMAL_TEST_H5, "traces/normal/test")
        convert_h5_to_traces(ATTACK_TRAIN_H5, "traces/attack/train")
        convert_h5_to_traces(ATTACK_VALID_H5, "traces/attack/validation")
        convert_h5_to_traces(ATTACK_TEST_H5, "traces/attack/test")
    if args.preprocess:
        logging.info("Preprocessing traces to graphs...")
        preprocess_traces_to_graphs()
    if args.train:
        logging.info("Training GNN model...")
        train_gnn_model()