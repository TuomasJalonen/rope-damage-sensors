#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for models.
Author: Tuomas Jalonen
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
# import tensorflow as tf
# from tensorflow_addons.metrics import F1Score
from tensorflow.keras.metrics import Precision, Recall, F1Score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, roc_curve, auc

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
    avg_metrics,
)

# Default values
DEFAULT_EPOCHS = 150
DEFAULT_BATCH_SIZE = 32

# Global paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"
RESULTS_PATH = PROJECT_ROOT / "results"
TRAIN_DATA_PATH = DATA_PATH / "train"
TEST_DATA_PATH = DATA_PATH / "test"

# Configure logging
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


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train models with configurable parameters."
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
    parser.add_argument(
        "--folds",
        type=str,
        help="Comma-separated list of folds to train (e.g., '1,3'). If not specified, all folds will be trained.",
    )
    return parser.parse_args()


def get_model(name, input_size, number_of_blocks):
    """Load model based on the provided name."""
    if name.startswith("Zhou_2019"):
        return zhou_2019_model(input_size)
    elif name.startswith("Zhou_2021"):
        return zhou_2021_model(input_size)
    elif name.startswith("kDenseNet_BC_L100_12ch"):
        return kDenseNet_BC_L100_12ch_model(input_size)
    else:
        return cnn_model(input_size, number_of_blocks)


def train_and_evaluate(args, model_config, folds_to_train):
    """Train and evaluate the selected model on specified folds."""
    model_name = args.model
    epochs = args.epochs
    batch_size = args.batch_size

    model_dir = RESULTS_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_img)

    # Correctly initialize metrics
    fold_metrics = initialize_metrics_dict()

    model = get_model(
        model_name, model_config["input_size"], model_config["number_of_blocks"]
    )
    model.compile(
        optimizer=Adam(learning_rate=lr_schedule(0)),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            # F1Score(num_classes=2, average="micro", name="f1_score"),
            F1Score(average="micro", name="f1_score"),
        ],
    )

    # Train and evaluate each selected fold
    for fold in folds_to_train:
        logging.info(f"Training {model_name}, Fold {fold}...")

        fold_dir = model_dir / f"Fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_gen = train_datagen.flow_from_directory(
            TRAIN_DATA_PATH / f"Fold_{fold}" / "Train",
            target_size=model_config["target_size"],
            batch_size=batch_size,
            color_mode=model_config["color_mode"],
            class_mode="categorical",
        )

        val_gen = val_datagen.flow_from_directory(
            TRAIN_DATA_PATH / f"Fold_{fold}" / "Validation",
            target_size=model_config["target_size"],
            batch_size=batch_size,
            color_mode=model_config["color_mode"],
            class_mode="categorical",
            shuffle=False,
        )

        test_gen = test_datagen.flow_from_directory(
            TEST_DATA_PATH,
            target_size=model_config["target_size"],
            batch_size=batch_size,
            color_mode=model_config["color_mode"],
            class_mode="categorical",
            shuffle=False,
        )

        # Train and evaluate the current fold
        metrics = train_and_evaluate_fold(
            model, fold_dir, train_gen, val_gen, test_gen, epochs, batch_size
        )

        # Store fold results for averaging later
        for key in fold_metrics:
            for metric_name, value in metrics[key].items():
                fold_metrics[key][metric_name].append(value)

    # Average metrics across selected folds
    avg_val_metrics = avg_metrics(
        fold_metrics["val"]["accuracy"],
        fold_metrics["val"]["precision"],
        fold_metrics["val"]["recall"],
        fold_metrics["val"]["f1_score"],
        fold_metrics["val"]["false_positive_rate"],
        fold_metrics["val"]["false_negative_rate"],
        fold_metrics["val"]["training_time"],
    )

    avg_test_metrics = avg_metrics(
        fold_metrics["test"]["accuracy"],
        fold_metrics["test"]["precision"],
        fold_metrics["test"]["recall"],
        fold_metrics["test"]["f1_score"],
        fold_metrics["test"]["false_positive_rate"],
        fold_metrics["test"]["false_negative_rate"],
        fold_metrics["test"]["training_time"],
    )

    # Save averaged metrics
    save_average_metrics(model_dir, avg_val_metrics, avg_test_metrics)


def train_and_evaluate_fold(
    model, fold_dir, train_gen, val_gen, test_gen, epochs, batch_size
):
    """Train and evaluate the model on one fold."""
    lr_scheduler = LearningRateScheduler(lr_schedule)
    acc_callback = save_best_acc(fold_dir)
    timing_callback = TimingCallback()

    # Train the model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        steps_per_epoch=train_gen.samples // batch_size,
        callbacks=[lr_scheduler, acc_callback, timing_callback],
    )

    # Save training history and learning curves
    np.save(fold_dir / "history.npy", history.history)
    pd.DataFrame(history.history).to_json(fold_dir / "history.json")
    plot_training_curves(fold_dir, history)

    # Load best model
    model.load_weights(fold_dir / "saved_model.weights.h5")

    # Evaluate predictions
    val_predictions = model.predict(val_gen)
    test_predictions = model.predict(test_gen)

    val_predicted_indices = np.argmax(val_predictions, axis=-1)
    val_true_indices = val_gen.classes

    test_predicted_indices = np.argmax(test_predictions, axis=-1)
    test_true_indices = test_gen.classes

    # Confusion matrices
    val_cm = confusion_matrix(val_true_indices, val_predicted_indices)
    test_cm = confusion_matrix(test_true_indices, test_predicted_indices)

    np.save(fold_dir / "val_cm.npy", val_cm)
    np.save(fold_dir / "test_cm.npy", test_cm)

    # Save confusion matrices
    val_df_cm = pd.DataFrame(
        val_cm, index=val_gen.class_indices.keys(), columns=val_gen.class_indices.keys()
    )
    test_df_cm = pd.DataFrame(
        test_cm,
        index=test_gen.class_indices.keys(),
        columns=test_gen.class_indices.keys(),
    )

    plot_cm(fold_dir, test_df_cm)

    # Save predictions
    pd.DataFrame(val_predictions).to_json(fold_dir / "val_predictions.json")
    pd.DataFrame(test_predictions).to_json(fold_dir / "test_predictions.json")

    # Save misclassified filenames
    misclassified_indices = np.where(test_predicted_indices != test_true_indices)[0]
    misclassified_filenames = np.array(test_gen.filenames)[misclassified_indices]
    np.save(fold_dir / "misclassified_filenames.npy", misclassified_filenames)
    np.savetxt(
        fold_dir / "misclassified_filenames.csv", misclassified_filenames, fmt="%s"
    )

    # Calculate metrics
    val_metrics = split_metrics(val_cm)
    test_metrics = split_metrics(test_cm)

    # Save metrics with labels
    save_fold_metrics(fold_dir, val_metrics, test_metrics, timing_callback.logs)

    # Generate and save ROC curves
    fpr, tpr, _ = roc_curve(test_true_indices, test_predictions[:, -1])
    AUC = auc(fpr, tpr)
    plot_auc(fold_dir, fpr, tpr, AUC)

    return {
        "val": {
            "accuracy": val_metrics[0],
            "precision": val_metrics[1],
            "recall": val_metrics[2],
            "f1_score": val_metrics[5],
            "false_positive_rate": val_metrics[3],
            "false_negative_rate": val_metrics[4],
            "training_time": timing_callback.logs,
        },
        "test": {
            "accuracy": test_metrics[0],
            "precision": test_metrics[1],
            "recall": test_metrics[2],
            "f1_score": test_metrics[5],
            "false_positive_rate": test_metrics[3],
            "false_negative_rate": test_metrics[4],
            "training_time": timing_callback.logs,
        },
    }


def evaluate_fold_metrics(model, fold_dir, val_gen, test_gen):
    """Evaluate the model and return metrics for validation and test sets."""
    model.load_weights(fold_dir / "saved_model")

    val_predictions = model.predict(val_gen)
    test_predictions = model.predict(test_gen)

    val_cm = confusion_matrix(val_gen.classes, np.argmax(val_predictions, axis=-1))
    test_cm = confusion_matrix(test_gen.classes, np.argmax(test_predictions, axis=-1))

    # Calculate metrics
    val_metrics = split_metrics(val_cm)
    test_metrics = split_metrics(test_cm)

    return val_metrics, test_metrics


def save_fold_metrics(fold_dir, val_metrics, test_metrics, training_time):
    """Save fold-specific metrics with labels."""
    labeled_val_metrics = {
        "accuracy": val_metrics[0],
        "precision": val_metrics[1],
        "recall": val_metrics[2],
        "f1_score": val_metrics[3],
        "false_positive_rate": val_metrics[4],
        "false_negative_rate": val_metrics[5],
        "training_time": training_time,
    }

    labeled_test_metrics = {
        "accuracy": test_metrics[0],
        "precision": test_metrics[1],
        "recall": test_metrics[2],
        "f1_score": test_metrics[3],
        "false_positive_rate": test_metrics[4],
        "false_negative_rate": test_metrics[5],
        "training_time": training_time,
    }

    # Save labeled metrics to JSON
    with open(fold_dir / "val_metrics.json", "w") as f:
        json.dump(labeled_val_metrics, f, indent=4)

    with open(fold_dir / "test_metrics.json", "w") as f:
        json.dump(labeled_test_metrics, f, indent=4)


def save_average_metrics(model_dir, avg_val_metrics, avg_test_metrics):
    """Save averaged metrics after all folds are completed."""
    with open(model_dir / "avg_val_metrics.json", "w") as f:
        json.dump(avg_val_metrics, f, indent=4)

    with open(model_dir / "avg_test_metrics.json", "w") as f:
        json.dump(avg_test_metrics, f, indent=4)

    logging.info("Averaged metrics saved successfully.")


def main():
    """Main function."""
    args = parse_arguments()

    # Determine models to train
    models_to_train = [args.model] if args.model else MODELS.keys()
    logging.info(f"Models to be trained: {models_to_train}")

    # Determine folds to train
    if args.folds:
        folds_to_train = [int(f.strip()) for f in args.folds.split(",")]
    else:
        folds_to_train = range(1, 5)  # Default: Train all folds

    logging.info(f"Folds to be trained: {folds_to_train}")

    # Train selected models on selected folds
    for model_name in models_to_train:
        if model_name not in MODELS:
            logging.error(
                f"Model '{model_name}' not found in available models. Skipping."
            )
            continue

        model_config = MODELS[model_name]
        logging.info(f"Starting training for model: {model_name}")

        train_and_evaluate(args, model_config, folds_to_train)
        logging.info(f"Training completed for model: {model_name}")

    logging.info("All models and folds have been trained.")


def initialize_metrics_dict():
    """Initialize fold metrics with correct labels."""
    return {
        "val": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "false_positive_rate": [],
            "false_negative_rate": [],
            "training_time": [],
        },
        "test": {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "false_positive_rate": [],
            "false_negative_rate": [],
            "training_time": [],
        },
    }


if __name__ == "__main__":
    main()
