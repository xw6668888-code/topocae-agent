"""Plot recorded R/Q training and validation losses for all three seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator


HERE = Path(__file__).resolve().parent
SOURCE_INDEX = HERE / "data/training_curves/index.json"
OUTPUT_STEM = HERE / "results/reproduced_figures/Fig3_training_three_seeds"
SEEDS = (20260718, 20260719, 20260720)
BRANCHES = ("R", "Q")


def load_curves():
    source_records = json.loads(SOURCE_INDEX.read_text(encoding="utf-8"))
    expected = {(branch, seed) for branch in BRANCHES for seed in SEEDS}
    actual = [(record["branch"], int(record["seed"])) for record in source_records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("The source index must contain all six distinct runs")
    curves = {}
    provenance = []
    for record in source_records:
        history_path = HERE / record["history"]
        history_bytes = history_path.read_bytes()
        digest = hashlib.sha256(history_bytes).hexdigest()
        if digest.lower() != record["sha256"].lower():
            raise ValueError(f"History hash mismatch: {history_path}")
        history = json.loads(history_bytes.decode("utf-8"))
        rows = [
            {
                "epoch": int(item["epoch"]),
                "training_loss": float(item["train"]["loss"]),
                "validation_loss": float(item["val"]["loss"]),
            }
            for item in history
        ]
        if len(rows) != int(record["epochs"]):
            raise ValueError(f"Epoch count differs from the source index: {history_path}")
        if [row["epoch"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Epoch sequence is not contiguous: {history_path}")
        if any(
            not math.isfinite(row[field]) or row[field] < 0
            for row in rows
            for field in ("training_loss", "validation_loss")
        ):
            raise ValueError(f"Invalid recorded loss: {history_path}")
        branch, seed = record["branch"], int(record["seed"])
        curves[branch, seed] = rows
        provenance.append(
            {
                "branch": branch,
                "seed": seed,
                "history": str(history_path),
                "sha256": digest,
                "epochs": len(rows),
                "first_training_loss": rows[0]["training_loss"],
                "last_training_loss": rows[-1]["training_loss"],
                "first_validation_loss": rows[0]["validation_loss"],
                "last_validation_loss": rows[-1]["validation_loss"],
            }
        )
    return curves, provenance


def main():
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    curves, provenance = load_curves()
    tnr_file = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.9), sharex=True, sharey=True)
    panel_map = []
    for row_index, branch in enumerate(BRANCHES):
        for column_index, seed in enumerate(SEEDS):
            axis = axes[row_index, column_index]
            rows = curves[branch, seed]
            epochs = [row["epoch"] for row in rows]
            axis.plot(
                epochs,
                [row["training_loss"] for row in rows],
                color="#222222",
                linewidth=1.8,
                label="Train loss",
            )
            axis.plot(
                epochs,
                [row["validation_loss"] for row in rows],
                color="#E41A1C",
                linewidth=1.8,
                linestyle=(0, (4, 1.8)),
                label="Validation loss",
            )
            panel_label = chr(ord("a") + row_index * 3 + column_index)
            axis.set_title(f"({panel_label}) {branch}, seed {seed}", loc="left", pad=9)
            axis.set_xlabel("Epoch")
            if column_index == 0:
                axis.set_ylabel("Multi-task loss")
            axis.set_xlim(0, 57)
            axis.set_xticks((0, 10, 20, 30, 40, 50))
            axis.set_ylim(0, 2.8)
            axis.set_yticks((0, 0.5, 1.0, 1.5, 2.0, 2.5))
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
            axis.tick_params(
                axis="both", which="major", direction="in",
                bottom=True, left=True, top=False, right=False,
                labelbottom=True, labelleft=True, length=5.5, width=1.0,
            )
            axis.tick_params(
                axis="both", which="minor", direction="in",
                bottom=True, left=True, top=False, right=False,
                length=3.0, width=0.8,
            )
            for spine in axis.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#222222")
            axis.legend(frameon=False, loc="upper right", handlelength=2.5)
            panel_map.append(
                {"panel": panel_label, "branch": branch, "seed": seed, "epochs": len(rows)}
            )
    fig.tight_layout(w_pad=1.8, h_pad=2.0)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            OUTPUT_STEM.with_suffix(f".{extension}"),
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)

    with OUTPUT_STEM.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("branch", "seed", "epoch", "training_loss", "validation_loss"),
        )
        writer.writeheader()
        for branch in BRANCHES:
            for seed in SEEDS:
                for row in curves[branch, seed]:
                    writer.writerow({"branch": branch, "seed": seed, **row})
    record = {
        "artifact": "Fig. 3: three-seed R/Q training and validation losses",
        "purpose": "Show actual loss trajectories and their variation across initializations",
        "manuscript_placement": "Section 5.1 Model training",
        "layout": "2 rows of branches by 3 columns of seeds",
        "source_index": SOURCE_INDEX.name,
        "font": "Times New Roman",
        "resolved_font_file": tnr_file,
        "tick_style": "inward; bottom and left only",
        "raw_values": True,
        "smoothing": False,
        "epoch_padding": False,
        "seed_aggregation": False,
        "png_dpi": 600,
        "curve_rows": sum(len(rows) for rows in curves.values()),
        "panel_map": panel_map,
        "sources": provenance,
    }
    OUTPUT_STEM.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"curves": panel_map, "curve_rows": record["curve_rows"]}))


if __name__ == "__main__":
    main()
