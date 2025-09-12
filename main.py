import os
import h5py
import pickle
import logging
import argparse
import subprocess
import multiprocessing
from typing import Dict
from data.dataset import load_dataset
from preprocessing.graph_preprocess_dataset import preprocess_dataset

NORMAL_TRAIN_H5 = os.getenv("NORMAL_TRAIN_H5", "Normal_DTDS-train.h5")
NORMAL_VALID_H5 = os.getenv("NORMAL_VALID_H5", "Normal_DTDS-validation.h5")
NORMAL_TEST_H5  = os.getenv("NORMAL_TEST_H5",  "Normal_DTDS-test.h5")
ATTACK_TRAIN_H5 = os.getenv("ATTACK_TRAIN_H5", "Attach_DTDS-train.h5")
ATTACK_VALID_H5 = os.getenv("ATTACK_VALID_H5", "Attach_DTDS-validation.h5")
ATTACK_TEST_H5  = os.getenv("ATTACK_TEST_H5",  "Attach_DTDS-test.h5")

TRAIN_TRACES_DIR    = "traces_train"
INFER_TRACES_DIR    = "traces_infer"
PKL_TRACES_FILENAME = "processed_graphs.pkl"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end processing and training script for syscall graph classification")
    parser.add_argument("--extract", "-e", action="store_true", help="Run H5 to trace extraction")
    parser.add_argument("--preprocess_train", "-p", action="store_true",
                        help="Run trace to graph preprocessing for training the model")
    parser.add_argument("--preprocess_infer", "-i", action="store_true",
                        help="Run trace to graph preprocessing for inference with the trained model")
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

    syscall_map = parse_syscall_tbl(syscall_tbl_path)

    os.makedirs(output_dir, exist_ok=True)
    counter = len(os.listdir(output_dir))
    with h5py.File(h5_path, "r") as h5f:
        sequences = h5f["sequences"]
        for seq in sequences:
            syscall_names = [syscall_map.get(int(sid), f"unknown_{sid}") for sid in seq]
            if len(syscall_names) < 2:
                # skip sequences with one or zero syscalls as they cannot be
                # used to build a graph
                continue
            trace_path = os.path.join(output_dir, f"trace_{counter}.txt")
            logging.debug(f"Writing {trace_path} with {len(syscall_names)} syscalls")
            with open(trace_path, "w") as f:
                for name in syscall_names:
                    f.write(f"{name}(\n") # one per line with '(' to match expected format
            counter += 1

def preprocess_traces_to_graphs_train():
    output_filepath = preprocess_dataset(TRAIN_TRACES_DIR, False, False, PKL_TRACES_FILENAME)
    print(f"Graphs saved to {output_filepath}")

def preprocess_traces_to_graphs_infer():
    assert os.path.exists(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"), \
        "Training dataset not found, please run preprocessing for training first."
    _, _, vocab = load_dataset(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}")
    output_filepath = preprocess_dataset(
        INFER_TRACES_DIR, False, False, PKL_TRACES_FILENAME, vocab=vocab
    )
    print(f"Graphs saved to {output_filepath}")

def train_gnn_model():
    assert os.path.exists(f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}"), \
        "Training dataset not found, please run preprocessing for training first."
    subprocess.run([
        "python", "train_graph.py",
        "--dataset_path", f"{TRAIN_TRACES_DIR}/{PKL_TRACES_FILENAME}",
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
        def extract_normal():
            convert_h5_to_traces(NORMAL_TRAIN_H5, f"{TRAIN_TRACES_DIR}/normal")
            convert_h5_to_traces(NORMAL_VALID_H5, f"{TRAIN_TRACES_DIR}/normal")
            convert_h5_to_traces(NORMAL_TEST_H5,  f"{INFER_TRACES_DIR}/normal")
        def extract_attack():
            convert_h5_to_traces(ATTACK_TRAIN_H5, f"{TRAIN_TRACES_DIR}/attack")
            convert_h5_to_traces(ATTACK_VALID_H5, f"{TRAIN_TRACES_DIR}/attack")
            convert_h5_to_traces(ATTACK_TEST_H5,  f"{INFER_TRACES_DIR}/attack")
        normal_proc = multiprocessing.Process(target=extract_normal)
        attack_proc = multiprocessing.Process(target=extract_attack)
        normal_proc.start()
        attack_proc.start()
        normal_proc.join()
        attack_proc.join()
    if args.preprocess_train:
        logging.info("Preprocessing traces to graphs to train the model...")
        preprocess_traces_to_graphs_train()
    if args.preprocess_infer:
        logging.info("Preprocessing traces to graphs for inference with the trained model...")
        preprocess_traces_to_graphs_infer()
    if args.train:
        logging.info("Training GNN model...")
        train_gnn_model()