import shutil
import logging

from pathlib import Path

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def merge_images(merged_directory: Path | str, log: logging.Logger) -> None:
    if merged_directory.exists():
        log.info(f"{merged_directory.name} already exists")
        return

    log.info(f"Creating {merged_directory.name}")
    merged_directory.mkdir(exist_ok=True)
    for name in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        dirpath = merged_directory.with_name(name=name)
        for jpg_image in dirpath.glob("*.jpg"):
            shutil.copy(jpg_image, merged_directory / jpg_image.name)


def plot_random_image(df: pd.DataFrame, images_dir: Path | str, size=(224, 224)):
    assert images_dir.is_dir() and "image_id" in df.columns
    random_index = tf.random.shuffle(df.index)[0].numpy()
    image_path = Path(df.image_id[random_index])
    filename = images_dir.joinpath(image_path)
    contents = tf.io.read_file(filename.as_posix())
    image = tf.image.decode_jpeg(contents, channels=3)
    X = tf.image.resize(image, size=size) / 255.0
    plt.imshow(X)
    plt.title(df.dx[random_index])
    plt.axis("off")
