#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retraining script for models.
Author: Tuomas Jalonen
"""

import os
import json
import logging
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Import custom functions
from helper import (
    lr_schedule,
    save_best_acc,
    TimingCallback,
    preprocess_img,
    cnn_model,
    zhou_2019_model,
    zhou_2021_model,
    kDenseNet_BC_L100_12ch_model,
    plot_training_curves,
    plot_cm,
    plot_auc,
    split_metrics,
)

# Default values
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train" / "Fold_1"
TRAIN_DIR = DATA_PATH / "Train"
VAL_DIR = DATA_PATH / "Validation"
MERGED_TRAIN_DIR = PROJECT_ROOT / "data" / "combined_train"
TEST_DIR = PROJECT_ROOT / "data" / "test"
RESULTS_DIR = PROJECT_ROOT / "results" / "retraining"

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define all models
MODELS = {
    # CNN Models
    "CNN1": {
        "input_size": (16, 16, 1),
        "target_size": (16, 16),
        "color_mode": "grayscale",
        "number_of_blocks": 1,
    },
    "CNN2": {
        "input_size": (16, 16, 1),
        "target_size": (16, 16),
        "color_mode": "grayscale",
        "number_of_blocks": 2,
    },
    "CNN3": {
        "input_size": (16, 16, 3),
        "target_size": (16, 16),
        "color_mode": "rgb",
        "number_of_blocks": 1,
    },
    "CNN4": {
        "input_size": (16, 16, 3),
        "target_size": (16, 16),
        "color_mode": "rgb",
        "number_of_blocks": 2,
    },
    "CNN5": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 1,
    },
    "CNN6": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 2,
    },
    "CNN7": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 3,
    },
    "CNN8": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 1,
    },
    "CNN9": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 2,
    },
    "CNN10": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 3,
    },
    "CNN11": {
        "input_size": (64, 64, 1),
        "target_size": (64, 64),
        "color_mode": "grayscale",
        "number_of_blocks": 1,
    },
    "CNN12": {
        "input_size": (64, 64, 1),
        "target_size": (64, 64),
        "color_mode": "grayscale",
        "number_of_blocks": 2,
    },
    "CNN13": {
        "input_size": (64, 64, 1),
        "target_size": (64, 64),
        "color_mode": "grayscale",
        "number_of_blocks": 3,
    },
    "CNN14": {
        "input_size": (64, 64, 3),
        "target_size": (64, 64),
        "color_mode": "rgb",
        "number_of_blocks": 1,
    },
    "CNN15": {
        "input_size": (64, 64, 3),
        "target_size": (64, 64),
        "color_mode": "rgb",
        "number_of_blocks": 2,
    },
    "CNN16": {
        "input_size": (64, 64, 3),
        "target_size": (64, 64),
        "color_mode": "rgb",
        "number_of_blocks": 3,
    },
    # Zhou 2019 Models
    "Zhou_2019_1": {
        "input_size": (16, 16, 1),
        "target_size": (16, 16),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2019_2": {
        "input_size": (16, 16, 3),
        "target_size": (16, 16),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "Zhou_2019_3": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2019_4": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "Zhou_2019_5": {
        "input_size": (64, 64, 1),
        "target_size": (64, 64),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2019_6": {
        "input_size": (64, 64, 3),
        "target_size": (64, 64),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    # Zhou 2021 Models
    "Zhou_2021_1": {
        "input_size": (16, 16, 1),
        "target_size": (16, 16),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2021_2": {
        "input_size": (16, 16, 3),
        "target_size": (16, 16),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "Zhou_2021_3": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2021_4": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "Zhou_2021_5": {
        "input_size": (64, 64, 1),
        "target_size": (64, 64),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "Zhou_2021_6": {
        "input_size": (64, 64, 3),
        "target_size": (64, 64),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "Zhou_2021_7": {
        "input_size": (96, 96, 1),
        "target_size": (96, 96),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    # DenseNet Models
    "kDenseNet_BC_L100_12ch_1": {
        "input_size": (16, 16, 1),
        "target_size": (16, 16),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "kDenseNet_BC_L100_12ch_2": {
        "input_size": (16, 16, 3),
        "target_size": (16, 16),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
    "kDenseNet_BC_L100_12ch_3": {
        "input_size": (32, 32, 1),
        "target_size": (32, 32),
        "color_mode": "grayscale",
        "number_of_blocks": 0,
    },
    "kDenseNet_BC_L100_12ch_4": {
        "input_size": (32, 32, 3),
        "target_size": (32, 32),
        "color_mode": "rgb",
        "number_of_blocks": 0,
    },
}


# Argument parser
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain models with configurable parameters."
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name from the available models list. If not specified, all models will be trained.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for training (default: {DEFAULT_BATCH_SIZE}).",
    )
    return parser.parse_args()


def combine_data():
    """Combine training and validation data into one folder using file copying."""
    if MERGED_TRAIN_DIR.exists() and any(MERGED_TRAIN_DIR.iterdir()):
        logging.info(
            f"{MERGED_TRAIN_DIR} already exists and is not empty. Skipping data merge."
        )
        return

    MERGED_TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    # Combine class folders from Train and Validation
    for class_name in os.listdir(TRAIN_DIR):
        class_train = TRAIN_DIR / class_name
        class_val = VAL_DIR / class_name
        target_dir = MERGED_TRAIN_DIR / class_name

        if not class_train.is_dir() or not class_val.is_dir():
            logging.warning(f"Skipping non-directory: {class_train} or {class_val}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy files from Train and Validation to the combined folder
        for src_dir in [class_train, class_val]:
            logging.info(f"Processing {src_dir}...")

            files_copied = 0
            for file in os.listdir(src_dir):
                src_file = src_dir / file
                dest_file = target_dir / file

                if src_file.is_file():
                    shutil.copy2(src_file, dest_file)
                    files_copied += 1
                    logging.info(f"Copied {src_file} to {dest_file}")

            if files_copied == 0:
                logging.warning(
                    f"No files copied from {src_dir}. Check directory structure."
                )

    if not any(MERGED_TRAIN_DIR.iterdir()):
        logging.error(f"No files were combined into {MERGED_TRAIN_DIR}. Exiting.")
        raise RuntimeError(f"{MERGED_TRAIN_DIR} is empty after data combination.")


def get_model(model_name, input_size, number_of_blocks):
    """Load appropriate model."""
    if model_name.startswith("Zhou_2019"):
        return zhou_2019_model(input_size)
    elif model_name.startswith("Zhou_2021"):
        return zhou_2021_model(input_size)
    elif model_name.startswith("kDenseNet_BC_L100_12ch"):
        return kDenseNet_BC_L100_12ch_model(input_size)
    else:
        return cnn_model(input_size, number_of_blocks)


def train_and_evaluate(model_name, model_config, epochs, batch_size):
    """Train and evaluate the specified model."""
    model_dir = RESULTS_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Data Generators
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)

    train_gen = train_datagen.flow_from_directory(
        MERGED_TRAIN_DIR,
        target_size=model_config["target_size"],
        batch_size=batch_size,
        color_mode=model_config["color_mode"],
        class_mode="categorical",
    )

    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=model_config["target_size"],
        batch_size=batch_size,
        color_mode=model_config["color_mode"],
        class_mode="categorical",
        shuffle=False,
    )

    # Initialize and compile the model
    model = get_model(
        model_name, model_config["input_size"], model_config["number_of_blocks"]
    )

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule(0)),
        loss="categorical_crossentropy",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall")],
    )

    # Training settings
    lr_scheduler = LearningRateScheduler(lr_schedule)
    timing_callback = TimingCallback()

    # Train the model
    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=train_gen.samples // batch_size,
        callbacks=[lr_scheduler, timing_callback],
    )

    # Save training results
    np.save(model_dir / "history.npy", history.history)
    pd.DataFrame(history.history).to_json(model_dir / "history.json")

    # Save the final model after the last epoch
    model.save_weights(model_dir / "final_model.weights.h5")
    logging.info(f"Final model saved at {model_dir / 'final_model.weights.h5'}")

    # Evaluate predictions
    test_predictions = model.predict(test_gen)
    test_predicted_indices = np.argmax(test_predictions, axis=-1)
    test_true_indices = test_gen.classes

    # Confusion matrix
    test_cm = confusion_matrix(test_true_indices, test_predicted_indices)
    test_df_cm = pd.DataFrame(
        test_cm,
        index=test_gen.class_indices.keys(),
        columns=test_gen.class_indices.keys(),
    )

    # Save confusion matrix and predictions
    np.save(model_dir / "test_cm.npy", test_cm)
    pd.DataFrame(test_predictions).to_json(model_dir / "test_predictions.json")

    # Save misclassified files
    misclassified_indices = np.where(test_predicted_indices != test_true_indices)[0]
    misclassified_filenames = np.array(test_gen.filenames)[misclassified_indices]
    np.save(model_dir / "misclassified_filenames.npy", misclassified_filenames)
    np.savetxt(
        model_dir / "misclassified_filenames.csv", misclassified_filenames, fmt="%s"
    )

    # Calculate metrics
    test_metrics = split_metrics(test_cm)
    fpr, tpr, _ = roc_curve(test_true_indices, test_predictions[:, -1])
    AUC = auc(fpr, tpr)

    # Save metrics
    test_metrics_dict = {
        "accuracy": test_metrics[0],
        "precision": test_metrics[1],
        "recall": test_metrics[2],
        "f1_score": test_metrics[5],
        "false_positive_rate": test_metrics[3],
        "false_negative_rate": test_metrics[4],
    }
    with open(model_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics_dict, f, indent=4)

    plot_cm(model_dir, test_df_cm)
    plot_auc(model_dir, fpr, tpr, AUC)

    logging.info(f"Training and evaluation complete for {model_name}.")


def main():
    """Main function."""
    args = parse_arguments()

    combine_data()

    # Determine models to train
    models_to_train = [args.model] if args.model else MODELS.keys()
    logging.info(f"Models to be trained: {models_to_train}")

    for model_name in models_to_train:
        if model_name not in MODELS:
            logging.error(f"Model '{model_name}' not found. Skipping.")
            continue

        model_config = MODELS[model_name]
        logging.info(f"Starting training for model: {model_name}")
        train_and_evaluate(model_name, model_config, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
