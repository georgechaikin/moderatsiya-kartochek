import csv
import os
from pathlib import Path

import click
from transformers import BlipImageProcessor

from moderatsiya_kartochek.model_processing import load_ov_model, predict_class


@click.command("make_submission")
@click.argument("data-dir", type=click.Path(dir_okay=True, exists=True))
@click.argument("save-dir", type=click.Path(dir_okay=True, exists=True))
@click.option("--threshold", type=float, default=0.4, help="Пороговое значение.")
def make_submission(data_dir: str | os.PathLike, save_dir: str | os.PathLike, threshold: float = 0.4) -> None:
    """Сабмит результата в SAVE_DIR на основе изображений из DATA_DIR."""
    models_dir = Path(__file__).parent / "models"
    model_path = models_dir / "blip_image_classifier.xml"
    ov_model = load_ov_model(model_path)
    processor = BlipImageProcessor.from_pretrained(models_dir)

    image_paths = Path(data_dir).glob("*")
    save_path = Path(save_dir) / "submission.csv"
    with save_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "label_id"], delimiter="\t")
        writer.writeheader()
        for img_path in image_paths:
            class_id = predict_class(img_path, ov_model, processor, threshold)
            writer.writerow({"image_name": img_path.name, "label_id": class_id})
