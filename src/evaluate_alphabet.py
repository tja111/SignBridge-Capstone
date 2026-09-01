"""Evaluate the Alphabet Mode checkpoint and create paper-ready reports."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data import DETRData
from model import DETR
from utils.boxes import stacker


ALPHABET_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Alphabet Mode accuracy")
    parser.add_argument("--checkpoint", default="checkpoints/alphabet_model.pt")
    parser.add_argument("--data", default="data/test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DETRData(args.data, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=stacker,
                        shuffle=False, num_workers=0)
    model = DETR(num_classes=len(ALPHABET_CLASSES), pretrained_backbone=False,
                 verbose=False).to(device)
    model.load_pretrained(args.checkpoint, device=device)
    model.eval()

    matrix = np.zeros((len(ALPHABET_CLASSES), len(ALPHABET_CLASSES)), dtype=int)
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            probabilities = outputs["pred_logits"].softmax(-1)[..., :-1]
            query_scores, query_classes = probabilities.max(-1)
            top_queries = query_scores.argmax(-1)
            predictions = query_classes[
                torch.arange(len(targets), device=device), top_queries
            ].cpu().tolist()
            for target, predicted in zip(targets, predictions):
                for actual in target["labels"].tolist():
                    matrix[actual, predicted] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "alphabet_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual \\ Predicted", *ALPHABET_CLASSES])
        for label, row in zip(ALPHABET_CLASSES, matrix):
            writer.writerow([label, *row])

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    report_path = output_dir / "alphabet_accuracy_report.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Alphabet Mode: Accuracy Report\n\n")
        overall = (correct / total * 100) if total else 0.0
        report.write(f"**Overall accuracy:** {overall:.1f}% ({correct}/{total} test labels)\n\n")
        report.write("| Actual letter | Correct | Total | Accuracy | Precision | Recall | F1 | Most common confusion |\n")
        report.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for index, label in enumerate(ALPHABET_CLASSES):
            row, column = matrix[index], matrix[:, index]
            actual_total, predicted_total, true_positive = int(row.sum()), int(column.sum()), int(row[index])
            precision = true_positive / predicted_total if predicted_total else 0.0
            recall = true_positive / actual_total if actual_total else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            alternatives = row.copy()
            alternatives[index] = 0
            other_index = int(alternatives.argmax())
            confusion = "—" if alternatives[other_index] == 0 else f"{ALPHABET_CLASSES[other_index]} ({alternatives[other_index]})"
            accuracy = true_positive / actual_total * 100 if actual_total else 0.0
            report.write(
                f"| {label} | {true_positive} | {actual_total} | {accuracy:.1f}% | "
                f"{precision * 100:.1f}% | {recall * 100:.1f}% | {f1 * 100:.1f}% | {confusion} |\n"
            )
        report.write("\nRows are actual letters; columns are model predictions.\n")

    figure_path = output_dir / "alphabet_confusion_matrix.png"
    fig, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Test images")
    axis.set_xticks(range(len(ALPHABET_CLASSES)), ALPHABET_CLASSES)
    axis.set_yticks(range(len(ALPHABET_CLASSES)), ALPHABET_CLASSES)
    axis.set_xlabel("Predicted letter")
    axis.set_ylabel("Actual letter")
    axis.set_title("Alphabet Mode Confusion Matrix")
    maximum = matrix.max() if matrix.size else 0
    for row in range(len(ALPHABET_CLASSES)):
        for column in range(len(ALPHABET_CLASSES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center",
                      color="white" if maximum and matrix[row, column] > maximum / 2 else "black",
                      fontsize=6)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    print(f"Report: {report_path}")
    print(f"Matrix: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
