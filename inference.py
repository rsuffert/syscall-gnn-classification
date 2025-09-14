import torch
import numpy as np
from main import INFER_TRACES_DIR, PKL_TRACES_FILENAME
from torch_geometric.data import DataLoader
from data.dataset import load_dataset, CustomGraphDataset
from models.models import GNNModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

TRAINED_MODEL_FILEPATH = "experiments/GIN/best_model_checkpoint.pt"
MALWARE_THRESHOLD = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

graph_data, vocab_size, _  = load_dataset(f"{INFER_TRACES_DIR}/{PKL_TRACES_FILENAME}")

labels = ["malware" if data["label"] == "attack" else data["label"] for data in graph_data]
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
checkpoint = torch.load(TRAINED_MODEL_FILEPATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
model.to(device)

all_probs = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        probs = torch.softmax(out, dim=1)
        all_probs.extend(probs.cpu().numpy())
all_probs_np = np.array(all_probs)

# Predict normal (class 1) if malware probability <= threshold, else malware (class 0)
threshold_preds = (all_probs_np[:, 0] <= MALWARE_THRESHOLD).astype(int)
decoded_preds = label_encoder.inverse_transform(threshold_preds)

print("\n--- Per-Class Metrics ---")
print(classification_report(labels, decoded_preds, target_names=label_encoder.classes_, digits=4))

print("\n--- Confusion Matrix ---")
cm = confusion_matrix(labels, decoded_preds, labels=label_encoder.classes_)
print("Confusion Matrix:")
print(f"Predicted:  {' '.join(f'{cls:>10}' for cls in label_encoder.classes_)}")
for i, cls in enumerate(label_encoder.classes_):
    print(f"True {cls:>7}: {' '.join(f'{cm[i][j]:>10}' for j in range(len(label_encoder.classes_)))}")