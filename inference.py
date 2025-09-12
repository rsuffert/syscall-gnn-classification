import torch
import numpy as np
from main import INFER_TRACES_DIR, PKL_TRACES_FILENAME
from torch_geometric.data import DataLoader
from data.dataset import load_dataset, CustomGraphDataset
from models.models import GNNModel
from sklearn.preprocessing import LabelEncoder

TRAINED_MODEL_FILEPATH = "experiments/GIN/best_model_checkpoint.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_data, vocab_size, _  = load_dataset(f"{INFER_TRACES_DIR}/{PKL_TRACES_FILENAME}")

labels = [data["label"] for data in graph_data]
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(["malware", "normal"])

graphs = [data["graph"] for data in graph_data]
dataset = CustomGraphDataset(graphs, len(label_encoder.classes_), training=False)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

model = GNNModel(
    vocab_size=vocab_size,
    embedding_dim=128,
    in_channels=dataset[0].num_node_features,
    hidden_channels=512,
    num_layers=4,
    out_channels=len(label_encoder.classes_),
    dropout=0.6,
    act="relu",
    model_type="GIN",
    heads=8,
    norm="batch"
)
checkpoint = torch.load(TRAINED_MODEL_FILEPATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

all_preds = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

decoded_preds = label_encoder.inverse_transform(all_preds)
total = 0
correct = 0
for i, pred in enumerate(decoded_preds):
    print(f"Trace {i}: Predicted class: {pred} | True class: {labels[i]}")
    total += 1
    if pred == labels[i]:
        correct += 1
print(f"Accuracy: {correct / total:.4f}")