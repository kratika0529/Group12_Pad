#!/usr/bin/env python3
"""
Compute APCER, BPCER, ACER for both trained models.
No retraining needed — loads saved checkpoints directly.
"""

import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ============================================================================
# CONFIGURATION — match exactly what was used in training
# ============================================================================
OUTPUT_DIR  = "dataset_clean"
BATCH_SIZE  = 32
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

CKPT_NO_AUG   = "No_Augmentation_best_1777825714.pth"
CKPT_WITH_AUG = "With_Augmentation_best_1777826692.pth"


# ============================================================================
# HELPERS
# ============================================================================
def get_test_loader():
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    test_dataset = datasets.ImageFolder(
        root=os.path.join(OUTPUT_DIR, "test"),
        transform=eval_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    return test_loader, test_dataset.classes


def load_model(checkpoint_path):
    model    = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE,
                                     weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, loader):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs  = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def compute_apcer_bpcer_acer(y_true, y_pred, classes):
    """
    APCER — Attack Presentation Classification Error Rate
            fraction of attack samples wrongly classified as real

    BPCER — Bona fide Presentation Classification Error Rate
            fraction of real samples wrongly classified as attack

    ACER  — Average Classification Error Rate = (APCER + BPCER) / 2
    """
    attack_idx = classes.index("attack")
    real_idx   = classes.index("real")

    attack_mask = (y_true == attack_idx)
    real_mask   = (y_true == real_idx)

    apcer = np.sum(y_pred[attack_mask] == real_idx)  / np.sum(attack_mask)
    bpcer = np.sum(y_pred[real_mask]   == attack_idx) / np.sum(real_mask)
    acer  = (apcer + bpcer) / 2.0

    return apcer, bpcer, acer


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 60)
    print("APCER / BPCER / ACER EVALUATION")
    print("=" * 60)
    print(f"Device: {DEVICE}\n")

    test_loader, classes = get_test_loader()
    print(f"Classes : {classes}")
    print(f"Test samples: {len(test_loader.dataset)}\n")

    results = {}

    for name, ckpt in [("No Augmentation",   CKPT_NO_AUG),
                        ("With Augmentation", CKPT_WITH_AUG)]:
        print(f"Loading {ckpt} ...")
        model = load_model(ckpt)
        preds, labels = get_predictions(model, test_loader)
        apcer, bpcer, acer = compute_apcer_bpcer_acer(labels, preds, classes)
        results[name] = (apcer, bpcer, acer)
        print(f"  Done — {len(labels)} samples evaluated")

    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<10} {'No Augmentation':>20} {'With Augmentation':>20}")
    print("-" * 52)
    for i, metric in enumerate(["APCER", "BPCER", "ACER"]):
        no_aug   = results["No Augmentation"][i]
        with_aug = results["With Augmentation"][i]
        print(f"{metric:<10} {no_aug:>20.4f} {with_aug:>20.4f}")
    print("-" * 52)

    print("\nDefinitions:")
    print("  APCER  — fraction of attacks passed as real (lower is better)")
    print("  BPCER  — fraction of real faces rejected as attack (lower is better)")
    print("  ACER   — (APCER + BPCER) / 2  (main ISO/IEC 30107-3 metric)")
    print()


if __name__ == "__main__":
    main()
