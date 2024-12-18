"""
This file contains utility functions.
"""

import os
import time
from skimage import exposure
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sn

import cai
from cai.densenet import ksimple_densenet


def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 120, 150 and 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    learning_rate = 1e-3
    if epoch > 180:
        learning_rate *= 1e-3
    elif epoch > 150:
        learning_rate *= 1e-2
    elif epoch > 120:
        learning_rate *= 1e-1
    print("Learning rate: ", learning_rate)
    return learning_rate


def save_best_acc(DIRNAME):
    acc_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(DIRNAME, "saved_model.weights.h5"),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    return acc_callback


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(time.time() - self.starttime)


def preprocess_img(img):
    img *= 1.0 / 255
    img_eq = exposure.equalize_hist(img)

    return img_eq


def cnn_model(INPUT_SHAPE, NUMBER_OF_BLOCKS):
    model = Sequential()
    # Preliminary
    model.add(
        Conv2D(
            64,
            (3, 3),
            activation="relu",
            input_shape=INPUT_SHAPE,
            kernel_regularizer=l2(5e-4),
        )
    )

    # First Block
    if NUMBER_OF_BLOCKS >= 1:
        model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.4))

    # Second Block
    if NUMBER_OF_BLOCKS >= 2:
        model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.4))

    # First Block
    if NUMBER_OF_BLOCKS >= 3:
        model.add(Conv2D(64, (3, 3), activation="relu", kernel_regularizer=l2(5e-4)))
        model.add(MaxPooling2D(2, 2))
        model.add(Dropout(0.4))

    # Fully Connected
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(20, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation="softmax"))

    return model


def zhou_2019_model(INPUT_SHAPE):
    model = Sequential(
        [
            Conv2D(
                32, (5, 5), activation="relu", padding="same", input_shape=INPUT_SHAPE
            ),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.5),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.5),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.5),
            Flatten(),
            Dropout(0.5),
            Dense(2560, activation="relu"),
            Dropout(0.5),
            Dense(768, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )

    return model


def zhou_2021_model(INPUT_SHAPE):
    model = Sequential(
        [
            Conv2D(
                16, (5, 5), activation="relu", padding="same", input_shape=INPUT_SHAPE
            ),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.5),
            Conv2D(96, (3, 3), activation="relu", padding="same"),
            MaxPooling2D(pool_size=(2, 2), strides=2),
            Dropout(0.5),
            Flatten(),
            Dropout(0.5),
            Dense(120, activation="relu"),
            Dropout(0.5),
            Dense(32, activation="relu"),
            Dropout(0.5),
            Dense(2, activation="softmax"),
        ]
    )

    return model


def kDenseNet_BC_L100_12ch_model(INPUT_SHAPE):
    model = ksimple_densenet(
        # [32, 32, 3],
        list(INPUT_SHAPE),
        # blocks=16,
        blocks=4,
        growth_rate=12,
        bottleneck=48,
        compression=0.5,
        l2_decay=0,
        kTypeTransition=cai.layers.D6_12ch(),
        kTypeBlock=cai.layers.D6_12ch(),
        num_classes=2,
        # dropout_rate=0.4,
        dropout_rate=0.3,
        activation=swish,
        has_interleave_at_transition=True,
    )

    return model


def plot_training_curves(dir, history):
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    # Accuracy curve
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig(os.path.join(dir, "acc.pdf"), dpi=300)
    plt.clf()

    # Loss curve
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Categorical Crossentropy")
    plt.xlabel("Epoch")
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig(os.path.join(dir, "loss.pdf"), dpi=300)
    plt.clf()

    # F1-score curve
    plt.plot(history.history["f1_score"])
    plt.plot(history.history["val_f1_score"])
    plt.ylabel("F1-score")
    plt.xlabel("Epoch")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig(os.path.join(dir, "f1.pdf"), dpi=300)
    plt.clf()

    # Precision curve
    plt.plot(history.history["precision"])
    plt.plot(history.history["val_precision"])
    plt.ylabel("Precision")
    plt.xlabel("Epoch")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig(os.path.join(dir, "precision.pdf"), dpi=300)
    plt.clf()

    # Recall curve
    plt.plot(history.history["recall"])
    plt.plot(history.history["val_recall"])
    plt.ylabel("Recall")
    plt.xlabel("Epoch")
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.legend(["Training", "Validation"], loc="upper left")
    plt.savefig(os.path.join(dir, "recall.pdf"), dpi=300)
    plt.clf()

    # Combined Accuracy and Loss curve
    fig, ax1 = plt.subplots()

    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_yticks(np.arange(0.50, 1.05, 0.05))
    ax1.set_ylim(0.50, 1.0)
    ax1.grid()

    ax2 = ax1.twinx()
    ax2.plot(history.history["loss"], "--")
    ax2.plot(history.history["val_loss"], "--")
    ax2.set_ylabel("Categorical Crossentropy")
    ax2.set_yticks(np.arange(0.0, 1.1, 0.1))
    ax2.set_ylim(0.0, 1.0)

    fig.legend(
        ["Training Accuracy", "Testing Accuracy", "Training Loss", "Testing Loss"],
        bbox_to_anchor=(0.9, 0.5),
    )
    plt.savefig(os.path.join(dir, "acc_loss.pdf"), dpi=300)
    plt.clf()

    return None


def plot_cm(dir, df_cm):
    sn.set_theme(font_scale=1.4)  # for label size
    sn.heatmap(df_cm, annot=True, fmt="d", annot_kws={"size": 16}, cmap="YlGnBu")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "cm.pdf"), dpi=300)
    plt.clf()

    return None


def plot_auc(dir, fpr, tpr, auc):
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc))
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "roc.pdf"), dpi=300)
    plt.clf()

    return None


def plot_tsne(dir, tsne, true_classes):
    # Restore Matplotlib settings
    matplotlib.rc_file_defaults()

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = np.max(x) - np.min(x)

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    sn.scatterplot(x=tx, y=ty, hue=true_classes, legend="full")
    plt.tight_layout()
    plt.savefig(os.path.join(dir, "t-sne.pdf"), dpi=300)
    plt.clf()

    return None


def split_metrics(cm):
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    f1 = 2 * tp / (2 * tp + fp + fn)

    return accuracy, precision, tpr, fpr, fnr, f1


import numpy as np


def avg_metrics(
    accuracies,
    precisions,
    recalls,
    f1_scores,
    false_positive_rates,
    false_negative_rates,
    training_times,
):
    """Calculate average metrics and standard deviations."""

    avg_metrics_dict = {
        "accuracy": {
            "mean": np.mean(accuracies),
            "std": np.std(accuracies),
            "individual": accuracies,
        },
        "precision": {
            "mean": np.mean(precisions),
            "std": np.std(precisions),
            "individual": precisions,
        },
        "recall": {
            "mean": np.mean(recalls),
            "std": np.std(recalls),
            "individual": recalls,
        },
        "f1_score": {
            "mean": np.mean(f1_scores),
            "std": np.std(f1_scores),
            "individual": f1_scores,
        },
        "false_positive_rate": {
            "mean": np.mean(false_positive_rates),
            "std": np.std(false_positive_rates),
            "individual": false_positive_rates,
        },
        "false_negative_rate": {
            "mean": np.mean(false_negative_rates),
            "std": np.std(false_negative_rates),
            "individual": false_negative_rates,
        },
        "average_training_time": {
            "mean": np.mean(training_times),
            "std": np.std(training_times),
            "individual": training_times,
        },
    }
    return avg_metrics_dict


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
