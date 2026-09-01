"""Create a per-class confusion report for the Words Mode checkpoint."""

import argparse
import csv
import json
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
from word_classes import WORD_CLASSES


def main():
    parser = argparse.ArgumentParser(description="Evaluate Words Mode per-class confusion")
    parser.add_argument("--checkpoint", default="checkpoints/words/words_model.pt")
    parser.add_argument("--data", default="data/words/test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    box_head_layers = 1
    words_image_size = 224
    meta_path = Path(args.checkpoint).with_name("meta.json")
    if meta_path.is_file():
        with meta_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        box_head_layers = int(metadata.get("box_head_layers", 1))
        words_image_size = int(metadata.get("words_image_size", 224))
    dataset = DETRData(args.data, train=False, words_mode=True,
                       words_image_size=words_image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=stacker,
                        shuffle=False, num_workers=0)
    model = DETR(num_classes=len(WORD_CLASSES), pretrained_backbone=False,
                 verbose=False, box_head_layers=box_head_layers).to(device)
    model.load_pretrained(args.checkpoint, device=device)
    model.eval()

    matrix = np.zeros((len(WORD_CLASSES), len(WORD_CLASSES)), dtype=int)
    with torch.no_grad():
        for images, targets in loader:
            outputs = model(images.to(device))
            probabilities = outputs["pred_logits"].softmax(-1)[..., :-1]
            query_scores, query_classes = probabilities.max(-1)
            top_queries = query_scores.argmax(-1)
            predictions = query_classes[torch.arange(len(targets), device=device), top_queries].cpu().tolist()
            for target, predicted in zip(targets, predictions):
                # Each test image normally has one sign. Record every annotated
                # target so rare multi-box images are not silently ignored.
                for actual in target["labels"].tolist():
                    matrix[actual, predicted] += 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "words_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual \\ Predicted", *WORD_CLASSES])
        for label, row in zip(WORD_CLASSES, matrix):
            writer.writerow([label, *row])

    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    report_path = output_dir / "words_confusion_report.md"
    with report_path.open("w", encoding="utf-8") as report:
        report.write("# Words Mode: Per-Class Confusion Report\n\n")
        report.write(f"**Overall accuracy:** {correct / total * 100:.1f}% ({correct}/{total} test labels)\n\n")
        report.write("| Actual sign | Correct | Total | Accuracy | Most common confusion |\n")
        report.write("|---|---:|---:|---:|---|\n")
        for index, label in enumerate(WORD_CLASSES):
            row = matrix[index]
            count = int(row.sum())
            accuracy = (row[index] / count * 100) if count else 0.0
            alternatives = row.copy()
            alternatives[index] = 0
            other_index = int(alternatives.argmax())
            confusion = "—" if alternatives[other_index] == 0 else f"{WORD_CLASSES[other_index]} ({alternatives[other_index]})"
            report.write(f"| {label} | {row[index]} | {count} | {accuracy:.1f}% | {confusion} |\n")
        report.write("\nRows are actual signs; columns are model predictions. "
                     "A high off-diagonal value identifies a pair of signs the model confuses.\n")

    figure_path = output_dir / "words_confusion_matrix.png"
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=axis, label="Test images")
    axis.set_xticks(range(len(WORD_CLASSES)), WORD_CLASSES, rotation=45, ha="right")
    axis.set_yticks(range(len(WORD_CLASSES)), WORD_CLASSES)
    axis.set_xlabel("Predicted sign")
    axis.set_ylabel("Actual sign")
    axis.set_title("Words Mode Confusion Matrix")
    for row in range(len(WORD_CLASSES)):
        for column in range(len(WORD_CLASSES)):
            axis.text(column, row, matrix[row, column], ha="center", va="center",
                      color="white" if matrix[row, column] > matrix.max() / 2 else "black", fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    print(f"Report: {report_path}")
    print(f"Matrix: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
