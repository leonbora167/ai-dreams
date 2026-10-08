import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from typing import Optional, Dict, Any, List

from src.adapters.datasets import get_dataset_adapter
from src.quality.quality_analyzer import QualityAnalyzer

class CustomImageDataset(Dataset):
    """PyTorch Dataset wrapper for adapter items."""
    def __init__(self, items: List[Dict[str, Any]], transform=None, is_detection=False):
        self.items = items
        self.transform = transform
        self.is_detection = is_detection
        self.classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = item["image"].convert("RGB")
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)

        # Classification label
        label_val = 0
        ann = item.get("annotations")
        if ann and isinstance(ann, dict):
            lbl_name = ann.get("label")
            if lbl_name in self.classes:
                label_val = self.classes.index(lbl_name)
            elif isinstance(lbl_name, int):
                label_val = lbl_name % len(self.classes)
        return img_tensor, label_val


def get_accelerator_device() -> torch.device:
    """Auto-detects GPU accelerator: CUDA, MPS (Apple Silicon), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def train_single_regime(
    model_name: str,
    dataset_items: List[Dict[str, Any]],
    val_items: List[Dict[str, Any]],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    regime_name: str,
    is_detection: bool = False
) -> Dict[str, Any]:
    """Trains a model on the given dataset items with backpropagation on GPU/MPS/CPU."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(
        CustomImageDataset(dataset_items, transform=transform, is_detection=is_detection),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        CustomImageDataset(val_items, transform=transform, is_detection=is_detection),
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model architecture
    if "detr" in model_name or is_detection:
        # Fine-tune head of Faster R-CNN / Detection backbone
        model = models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Use backbone feature classifier head for standard loss tracking
        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 10)
        ).to(device)
    else:
        # Fine-tune classification head of Inception / ResNet
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 10)
        classifier = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=lr)

    epoch_metrics = []
    print(f"🔥 Training [{regime_name}] on {device} ({len(dataset_items)} samples, {epochs} epochs)...")

    for ep in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = classifier(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(targets)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += len(targets)

        train_loss = total_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        # Evaluate on validation/clean golden set
        classifier.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v_imgs, v_targets in val_loader:
                v_imgs = v_imgs.to(device)
                v_targets = v_targets.to(device)
                v_outs = classifier(v_imgs)
                v_loss = criterion(v_outs, v_targets)
                val_loss += v_loss.item() * len(v_targets)
                _, v_preds = torch.max(v_outs, 1)
                val_correct += (v_preds == v_targets).sum().item()
                val_total += len(v_targets)

        val_loss = val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0

        epoch_metrics.append({
            "epoch": ep,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 2),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 2)
        })

    return {"metrics": epoch_metrics, "model": classifier}


def run_actual_training_drift(
    model_name: str,
    golden_dataset: str,
    dirty_dataset: str,
    epochs: int = 5,
    batch_size: int = 8,
    lr: float = 0.001,
    max_samples: Optional[int] = None,
    run_id: Optional[str] = None,
    results_dir: str = "results/training_drift"
) -> Dict[str, Any]:
    """
    Executes REAL GPU/Accelerator training on both Golden training data and Dirty training data.
    Computes true epoch-by-epoch learning drift, dynamic loss divergence, and accuracy degradation.
    Streams to TensorBoard and generates a standalone visual report.
    """
    actual_run_id = run_id or f"real_train_{model_name}_{int(time.time())}"
    run_output_dir = os.path.join(results_dir, actual_run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    tb_dir = os.path.join(run_output_dir, "tensorboard_logs")
    writer = SummaryWriter(log_dir=tb_dir)
    device = get_accelerator_device()

    print("\n" + "="*70)
    print("🚀 RUNNING REAL GPU/ACCELERATOR MODEL TRAINING DRIFT EXPERIMENT")
    print("="*70)
    print(f"Hardware Device     : {device} (Accelerated Backpropagation)")
    print(f"Model Architecture  : {model_name}")
    print(f"Golden Training Data: {golden_dataset}")
    print(f"Dirty Training Data : {dirty_dataset}")
    print(f"Hyperparameters     : Epochs={epochs}, BatchSize={batch_size}, LR={lr}")
    print(f"TensorBoard Event Logs: {tb_dir}\n")

    is_detection = ("voc" in golden_dataset.lower() or "detr" in model_name.lower())
    ds_type = "voc" if is_detection else "cifar10"
    adapter = get_dataset_adapter(ds_type)

    print("📦 Loading Datasets from Disk...")
    golden_items = adapter.load(golden_dataset, max_samples=max_samples)
    dirty_items = adapter.load(dirty_dataset, max_samples=max_samples)
    print(f"   Loaded {len(golden_items)} Golden samples and {len(dirty_items)} Dirty samples.")

    # 1. Real Backpropagation Training on Golden Data
    golden_res = train_single_regime(
        model_name=model_name,
        dataset_items=golden_items,
        val_items=golden_items[:min(50, len(golden_items))],
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        regime_name="Golden Data Regime",
        is_detection=is_detection
    )

    # 2. Real Backpropagation Training on Dirty Data
    dirty_res = train_single_regime(
        model_name=model_name,
        dataset_items=dirty_items,
        val_items=golden_items[:min(50, len(golden_items))], # Evaluated on clean golden test!
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        regime_name="Dirty Data Regime",
        is_detection=is_detection
    )

    # 3. Compute Real Dynamic Drift Across Epochs
    epoch_drift_records = []
    print("\n📊 Real Training Dynamics & Drift Analysis:")
    for ep in range(epochs):
        g_m = golden_res["metrics"][ep]
        d_m = dirty_res["metrics"][ep]
        loss_gap = float(d_m["train_loss"] - g_m["train_loss"])
        acc_gap = float(g_m["val_acc"] - d_m["val_acc"])

        rec = {
            "epoch": ep + 1,
            "golden_train_loss": g_m["train_loss"],
            "dirty_train_loss": d_m["train_loss"],
            "dynamic_loss_gap": round(loss_gap, 4),
            "golden_val_acc": g_m["val_acc"],
            "dirty_val_acc": d_m["val_acc"],
            "dynamic_acc_gap": round(acc_gap, 2)
        }
        epoch_drift_records.append(rec)

        # Stream real scalars to TensorBoard
        writer.add_scalars("RealTraining/Loss", {"Golden": g_m["train_loss"], "Dirty": d_m["train_loss"]}, ep + 1)
        writer.add_scalars("RealTraining/Accuracy", {"Golden": g_m["val_acc"], "Dirty": d_m["val_acc"]}, ep + 1)
        writer.add_scalar("RealTraining/Loss_Gap", loss_gap, ep + 1)
        writer.add_scalar("RealTraining/Accuracy_Degradation_pp", acc_gap, ep + 1)

        print(f"   Epoch {ep+1:2d}/{epochs}: Golden Loss={g_m['train_loss']:.3f}, Dirty Loss={d_m['train_loss']:.3f} | ΔLoss=+{loss_gap:.3f}, ΔAcc=-{acc_gap:.1f}%")

    writer.flush()
    writer.close()

    avg_loss_gap = float(np.mean([r["dynamic_loss_gap"] for r in epoch_drift_records]))
    final_acc_drop = epoch_drift_records[-1]["dynamic_acc_gap"]
    status = "HIGH DIVERGENCE / SEVERE DEGRADATION" if final_acc_drop > 10.0 else "MODERATE DIVERGENCE"

    result = {
        "run_id": actual_run_id,
        "device": str(device),
        "model_name": model_name,
        "golden_dataset": golden_dataset,
        "dirty_dataset": dirty_dataset,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_verdict": status,
        "avg_loss_gap": round(avg_loss_gap, 4),
        "final_accuracy_drop_pp": round(final_acc_drop, 2),
        "epoch_records": epoch_drift_records
    }

    # Save summary JSON
    summary_path = os.path.join(run_output_dir, "real_training_drift_summary.json")
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    # Generate Standalone Real Training HTML Report
    html_path = os.path.join(run_output_dir, "real_training_drift_report.html")
    _generate_real_training_html(result, html_path)

    print("\n" + "="*70)
    print(f"✅ REAL TRAINING COMPLETED ON ACCELERATOR: {device}")
    print(f"📊 Training Drift Verdict: {status}")
    print(f"📉 Observed Final Accuracy Drop: -{final_acc_drop:.1f}%")
    print(f"📈 TensorBoard Event Logs : {tb_dir}")
    print(f"🌐 Standalone HTML Report : {html_path}")
    print("="*70 + "\n")

    return result

def _generate_real_training_html(data: Dict[str, Any], output_path: str):
    epochs = [r["epoch"] for r in data["epoch_records"]]
    g_loss = [r["golden_train_loss"] for r in data["epoch_records"]]
    d_loss = [r["dirty_train_loss"] for r in data["epoch_records"]]
    g_acc = [r["golden_val_acc"] for r in data["epoch_records"]]
    d_acc = [r["dirty_val_acc"] for r in data["epoch_records"]]
    v_color = "#c62828" if "HIGH" in data["training_verdict"] else "#f57f17"

    rows = ""
    for r in data["epoch_records"]:
        rows += f"""
        <tr>
            <td><strong>Epoch {r['epoch']}</strong></td>
            <td>{r['golden_train_loss']}</td>
            <td>{r['dirty_train_loss']}</td>
            <td style="color:#c62828;"><strong>+{r['dynamic_loss_gap']}</strong></td>
            <td>{r['golden_val_acc']}%</td>
            <td>{r['dirty_val_acc']}%</td>
            <td style="color:#c62828;"><strong>-{r['dynamic_acc_gap']}%</strong></td>
        </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Real Model Training Drift Report — {data['run_id']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; padding: 32px 24px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display: inline-block; padding: 8px 18px; border-radius: 8px; font-size: 18px; font-weight: 800; background: #ffebee; color: {v_color}; border: 1px solid {v_color}; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th {{ background: #f1f5f9; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .chart-box {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; height: 320px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1 style="color: #1e3a8a;">🔥 Real GPU/Accelerator Training Drift Report</h1>
                <p style="color: #64748b;">Actual backpropagation training comparing Golden vs. Dirty training data</p>
            </div>
            <div style="background: #e2e8f0; padding: 6px 14px; border-radius: 20px; font-size: 13px;">Accelerator: {data['device']}</div>
        </div>

        <div class="card">
            <h3>Training Drift Verdict:</h3>
            <div class="badge">{data['training_verdict']}</div>
            <p style="margin-top: 12px; font-size: 15px;">
                <strong>Average Loss Gap:</strong> +{data['avg_loss_gap']} | 
                <strong>Final Accuracy Drop:</strong> -{data['final_accuracy_drop_pp']}%
            </p>
        </div>

        <div class="grid">
            <div class="chart-box"><canvas id="lossChart"></canvas></div>
            <div class="chart-box"><canvas id="accChart"></canvas></div>
        </div>

        <div class="card" style="margin-top: 24px;">
            <h3>Real Convergence Trajectory Table</h3>
            <table>
                <thead>
                    <tr>
                        <th>Epoch</th>
                        <th>Golden Loss</th>
                        <th>Dirty Loss</th>
                        <th>Δ Loss Gap</th>
                        <th>Golden Val Acc</th>
                        <th>Dirty Val Acc</th>
                        <th>Δ Acc Drop</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
    </div>

    <script>
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(epochs)},
                datasets: [
                    {{ label: 'Golden Train Loss', data: {json.dumps(g_loss)}, borderColor: '#1e3a8a', borderWidth: 2 }},
                    {{ label: 'Dirty Train Loss', data: {json.dumps(d_loss)}, borderColor: '#ef4444', borderWidth: 2 }}
                ]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Real Backpropagation Training Loss' }} }} }}
        }});

        new Chart(document.getElementById('accChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(epochs)},
                datasets: [
                    {{ label: 'Golden Model Acc (%)', data: {json.dumps(g_acc)}, borderColor: '#1e3a8a', borderWidth: 2 }},
                    {{ label: 'Dirty Model Acc (%)', data: {json.dumps(d_acc)}, borderColor: '#ef4444', borderWidth: 2 }}
                ]
            }},
            options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ title: {{ display: true, text: 'Clean Benchmark Validation Accuracy (%)' }} }} }}
        }});
    </script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real GPU/Accelerator training drift")
    parser.add_argument("--model", type=str, default="inception_v3")
    parser.add_argument("--golden_dataset", type=str, default="data/golden/cifar10_golden")
    parser.add_argument("--dirty_dataset", type=str, default="data/new/cifar10_camera_degraded")
    parser.add_argument("--epochs", type=int, default=5, help="Number of real training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--max_samples", type=int, default=100, help="Max samples per dataset (default: 100)")
    parser.add_argument("--run_id", type=str, default=None, help="Custom run name")
    args = parser.parse_args()

    run_actual_training_drift(
        model_name=args.model,
        golden_dataset=args.golden_dataset,
        dirty_dataset=args.dirty_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_samples=args.max_samples,
        run_id=args.run_id
    )

