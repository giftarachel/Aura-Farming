"""
Aura Farming — training/train_baseline.py
==========================================
MobileNetV3 Small baseline training script.

────────────────────────────────────────────────────────────────
PYTHON / TENSORFLOW VERSION REQUIREMENT
────────────────────────────────────────────────────────────────
TensorFlow officially supports Python 3.9–3.12 only.
This project's default .venv uses Python 3.14, which is NOT
supported by any current TensorFlow release.

You must run this script in one of these environments:

  Option A — Google Colab (recommended, free GPU):
    1. Upload the project to Google Drive or clone from GitHub.
    2. Open training/train_baseline.py in Colab.
    3. Colab uses Python 3.10 and includes TensorFlow.
    4. Download the .keras checkpoint when done.

  Option B — Local Python 3.12 environment:
    python3.12 -m venv .venv312
    source .venv312/bin/activate
    pip install -r requirements.txt
    # Apple Silicon: also run:
    pip install tensorflow-macos tensorflow-metal
    # Standard (Linux/Windows):
    pip install tensorflow
    python training/train_baseline.py

────────────────────────────────────────────────────────────────
USAGE (from project root)
────────────────────────────────────────────────────────────────
  # Phase A only (feature extraction with frozen base):
  python training/train_baseline.py

  # Phase A + Phase B (fine-tuning):
  python training/train_baseline.py --fine-tune

  # Override defaults:
  python training/train_baseline.py --batch-size 16 --epochs 20

  # Disable augmentation (faster but usually worse):
  python training/train_baseline.py --no-augment

────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES
────────────────────────────────────────────────────────────────
Phase A — Feature extraction:
  - Load MobileNetV3Small pretrained on ImageNet.
  - Freeze the entire convolutional base.
  - Train only the new classification head.
  - Monitor val_loss with early stopping (patience from config).
  - Save best weights to models/mobilenetv3_phase_a.keras.

Phase B — Fine-tuning (requires --fine-tune):
  - Load the Phase A checkpoint.
  - Unfreeze base layers from --fine-tune-at onward.
  - Continue training at a lower learning rate.
  - Save best weights to models/mobilenetv3_baseline.keras.

After each phase:
  - Evaluate on the VALIDATION set only.
  - Save per-class metrics and a confusion matrix to reports/.
  - The TEST set is never used here (reserved for Phase 7).

Class imbalance:
  - sklearn's 'balanced' class weights are computed from the
    training split only and passed to model.fit().

Preprocessing notes:
  - MobileNetV3's preprocess_input scales pixels from [0, 255]
    to [-1, 1] (re-scaling to the mobilenet_v3 expected range).
  - include_preprocessing=False is set on the base model to
    avoid double preprocessing.
  - The same preprocess_input function is used in src/preprocessing.py
    for inference so training and inference are identical.

Reproducibility:
  - PYTHONHASHSEED, NumPy, and TensorFlow seeds are all set from
    config.yaml before any model or data objects are created.

Phase 4 status: COMPLETE — ready for training in a compatible env.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import load_manifest


# ─────────────────────────────────────────────────────────────────
# Configuration helpers
# ─────────────────────────────────────────────────────────────────

def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_class_mapping() -> dict:
    mapping_path = PROJECT_ROOT / "configs" / "class_mapping.json"
    with open(mapping_path) as f:
        return json.load(f)


def set_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Must be called before importing any TensorFlow or Keras symbols.
    Sets Python hash seed, NumPy, and (once TF is imported) TF seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# ─────────────────────────────────────────────────────────────────
# TensorFlow import guard
# ─────────────────────────────────────────────────────────────────

def _import_tensorflow():
    """
    Import TensorFlow and set its random seed.

    Provides a clear error message when TensorFlow is not installed
    or the Python version is incompatible.
    """
    try:
        import tensorflow as tf
        return tf
    except ImportError:
        print(
            "\n❌  TensorFlow is not installed in this environment.\n"
            "\n   This project uses Python 3.14, which is NOT supported by\n"
            "   any current TensorFlow release (supported: Python 3.9–3.12).\n"
            "\n   To train the model, use one of these options:\n"
            "\n   Option A — Google Colab (recommended, free T4 GPU):\n"
            "     Upload the project and run training there.\n"
            "\n   Option B — Local Python 3.12 environment:\n"
            "     python3.12 -m venv .venv312\n"
            "     source .venv312/bin/activate\n"
            "     pip install -r requirements.txt\n"
            "     # Apple Silicon:\n"
            "     pip install tensorflow-macos tensorflow-metal\n"
            "     # Standard:\n"
            "     pip install tensorflow\n"
            "     python training/train_baseline.py\n"
        )
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────
# Manifest loading
# ─────────────────────────────────────────────────────────────────

def load_split_paths_and_labels(
    manifest_path: Path,
) -> tuple[list[str], list[int]]:
    """
    Load file paths and class labels from a split manifest CSV.

    The manifest paths are already absolute (written by prepare_dataset.py).
    No further resolution is needed.

    Args:
        manifest_path: Path to a manifest CSV file.

    Returns:
        Tuple of (paths, labels) where paths are absolute string paths
        and labels are integer class indices.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run training/prepare_dataset.py first."
        )
    records = load_manifest(manifest_path)
    paths  = [r["path"] for r in records]
    labels = [r["class_index"] for r in records]
    return paths, labels


def compute_class_weights(
    train_labels: list[int],
    num_classes: int,
) -> dict[int, float]:
    """
    Compute balanced class weights from TRAINING labels only.

    Uses sklearn's 'balanced' strategy:
        weight_i = n_samples / (n_classes × count_i)

    This penalises the majority class (Yellow Leaf Curl Virus, 3748
    training images) and upweights the minority class (Mosaic Virus,
    261 training images), giving the optimizer a fairer signal across
    all 10 disease categories.

    These weights are computed from the TRAINING manifest only.
    Validation and test sets are never used here.

    Args:
        train_labels: Integer class indices from the training split.
        num_classes: Total number of output classes.

    Returns:
        Dict mapping integer class index → float weight.
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.arange(num_classes)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(train_labels),
    )
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}

    print("\n  Class weights (computed from training split only):")
    for idx, w in sorted(weight_dict.items()):
        print(f"    Class {idx:>2}: {w:.4f}")
    print()

    return weight_dict


# ─────────────────────────────────────────────────────────────────
# tf.data pipeline
# ─────────────────────────────────────────────────────────────────

def build_tf_dataset(
    paths: list[str],
    labels: list[int],
    input_size: tuple[int, int],
    batch_size: int,
    augment: bool,
    shuffle: bool,
    seed: int,
):
    """
    Build a tf.data.Dataset of (image_tensor, label) pairs.

    Preprocessing applied here (training AND inference must match):
        1. Read file from disk.
        2. Decode JPEG/PNG → uint8 tensor, 3 channels (RGB).
        3. Resize to input_size (bilinear by default).
        4. Cast to float32.
        5. Manual MobileNetV3 rescaling: x / 127.5 - 1.0 → [-1, 1].
           (Not keras.applications.mobilenet_v3.preprocess_input, which
           became a no-op in Keras 3 / TF >= 2.16.)

    Augmentation (training only — never applied to val/test):
        - Random horizontal flip.
        - Random vertical flip.
        - Random 90° rotation (k ∈ {0,1,2,3}).
        - Random brightness jitter (±10%).
        - Random contrast jitter (0.9×–1.1×).
        - Values clipped back to [-1, 1] after jitter.

    Important: augmentation uses stateless random ops fed by a
    dataset-level global seed (tf.random.set_seed), not per-op
    seeds (which were deprecated in TF 2.x for tf.image ops).

    Args:
        paths: Absolute path strings for image files.
        labels: Integer class indices (same order as paths).
        input_size: (height, width) in pixels, e.g. (224, 224).
        batch_size: Images per batch.
        augment: Whether to apply training augmentation.
        shuffle: Whether to shuffle before each epoch.
        seed: Global dataset seed for reproducibility.

    Returns:
        A tf.data.Dataset yielding (batch_image, batch_label) pairs,
        batched and prefetched.
    """
    import tensorflow as tf

    h, w = input_size
    AUTOTUNE = tf.data.AUTOTUNE

    def _load_and_preprocess(path: tf.Tensor, label: tf.Tensor):
        raw = tf.io.read_file(path)
        # decode_image handles JPEG, PNG, WEBP, BMP automatically
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [h, w])
        img = tf.cast(img, tf.float32)
        # MobileNetV3 rescaling: [0, 255] → [-1, 1]
        # We apply this manually instead of using
        # keras.applications.mobilenet_v3.preprocess_input, which
        # became a no-op in Keras 3 (TF >= 2.16). The model is built
        # with include_preprocessing=False, so we must rescale here.
        img = img / 127.5 - 1.0
        return img, label

    def _augment(img: tf.Tensor, label: tf.Tensor):
        # Use stateless random ops — seed per-image from epoch counter
        # Flip left-right
        img = tf.image.random_flip_left_right(img)
        # Flip up-down (less common but adds diversity)
        img = tf.image.random_flip_up_down(img)
        # Rotate 90° × k, k uniform in {0,1,2,3}
        k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        # Subtle brightness / contrast jitter
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(_load_and_preprocess, num_parallel_calls=AUTOTUNE)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


# ─────────────────────────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    input_size: tuple[int, int],
    dropout_rate: float = 0.3,
) -> "tf.keras.Model":
    """
    Build MobileNetV3 Small with a custom 10-class head.

    Architecture:
        Input (224 × 224 × 3)
        └─ MobileNetV3Small (ImageNet weights, top removed, frozen)
           └─ GlobalAveragePooling2D
              └─ BatchNormalization
                 └─ Dropout(dropout_rate)
                    └─ Dense(num_classes, softmax)

    Why include_preprocessing=False and manual rescaling:
        We rescale pixels manually (x / 127.5 - 1.0) in the data
        pipeline (build_tf_dataset) and in src/preprocessing.py.
        Setting include_preprocessing=False prevents the base model
        from applying its own Rescaling layer, which would
        double-transform the input and degrade accuracy.

        Note: keras.applications.mobilenet_v3.preprocess_input became
        a no-op in Keras 3 (TF >= 2.16), so we cannot rely on it.
        The manual formula x / 127.5 - 1.0 is identical to what the
        internal Rescaling layer does when include_preprocessing=True.

    Args:
        num_classes: Number of output classes (10 for this project).
        input_size: (height, width), must be (224, 224).
        dropout_rate: Dropout probability in the classification head.

    Returns:
        Compiled (but not yet trained) Keras Model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import MobileNetV3Small
    from tensorflow.keras.optimizers import Adam

    h, w = input_size
    inputs = tf.keras.Input(shape=(h, w, 3), name="input_image")

    base = MobileNetV3Small(
        input_shape=(h, w, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=False,  # we rescale manually in the data pipeline
        name="mobilenetv3small",       # explicit name for get_layer() lookups
    )
    base.trainable = False  # Phase A: only the new head is trained

    # Pass training=False to the base so BatchNorm runs in inference
    # mode even when the overall model is in training mode. This is
    # important during Phase A when the base is frozen.
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(dropout_rate, name="dropout_head")(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           name="predictions")(x)

    model = Model(inputs, outputs, name="aura_farming_mobilenetv3")

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def unfreeze_for_fine_tuning(
    model: "tf.keras.Model",
    fine_tune_at: int,
    learning_rate: float = 1e-5,
) -> None:
    """
    Unfreeze the upper portion of the MobileNetV3Small base for Phase B.

    Freezes all base layers below `fine_tune_at`, unfreezes the rest,
    then re-freezes every BatchNormalization layer regardless of its
    position. BN layers must stay frozen during fine-tuning because
    their running statistics (mean, variance) were computed during
    ImageNet pretraining — updating them with a different dataset
    distribution can destabilize the model.

    Recompiles with a much lower learning rate to avoid large gradient
    updates that would overwrite the ImageNet features.

    Args:
        model: Keras model built by build_model().
        fine_tune_at: Index of the first base layer to unfreeze.
                      MobileNetV3Small has ~100+ layers; unfreezing from
                      index 80–100 is a typical starting point.
                      Must be in [0, len(base.layers)].
        learning_rate: Adam LR for fine-tuning (default 1e-5).

    Raises:
        ValueError: If fine_tune_at is outside [0, len(base.layers)].
    """
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    # The base is the MobileNetV3Small sub-model. We set its name
    # explicitly to "mobilenetv3small" in build_model() so this lookup
    # is version-independent (Keras 2 auto-lowercased it, Keras 3 preserves case).
    base = model.get_layer("mobilenetv3small")

    # ── Validate fine_tune_at ──────────────────────────────────────
    n_base_layers = len(base.layers)
    if fine_tune_at < 0 or fine_tune_at > n_base_layers:
        raise ValueError(
            f"fine_tune_at={fine_tune_at} is out of range. "
            f"MobileNetV3Small base has {n_base_layers} layers "
            f"(valid range: 0–{n_base_layers})."
        )

    base.trainable = True

    # Freeze all layers below fine_tune_at
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False

    # Keep all BatchNormalization layers frozen during fine-tuning.
    # BN layers contain running statistics (mean, variance) computed
    # during ImageNet pretraining. Updating these during fine-tuning
    # with a different dataset distribution can destabilize the model
    # and cause large accuracy drops. This is standard practice in
    # transfer learning — the BN moving statistics should reflect the
    # pretrained distribution, not the fine-tuning distribution.
    # We use isinstance() rather than name-string matching so that
    # any BN layer, regardless of its name, is reliably detected.
    bn_frozen_count = 0
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            bn_frozen_count += 1

    trainable_count = sum(1 for la in base.layers if la.trainable)
    total_count = n_base_layers
    print(
        f"\n  Fine-tuning: {trainable_count}/{total_count} base layers "
        f"unfrozen (from index {fine_tune_at})."
    )
    print(f"  BatchNormalization layers kept frozen: {bn_frozen_count}")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# ─────────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────────

def build_callbacks(
    checkpoint_path: Path,
    patience: int,
    phase_label: str,
) -> list:
    """
    Build the standard callback set for a training phase.

    Callbacks used:
        ModelCheckpoint  — saves the best epoch (lowest val_loss)
                           to checkpoint_path. Only the best is kept.
        EarlyStopping    — stops training when val_loss has not
                           improved for `patience` epochs, then
                           restores the best weights automatically.
        ReduceLROnPlateau — halves the learning rate when val_loss
                           has not improved for patience/2 epochs.
                           This helps escape flat loss plateaus.
        CSVLogger        — appends per-epoch train/val metrics to
                           reports/training_log_{phase_label}.csv
                           for later plotting and comparison.

    Args:
        checkpoint_path: Where to save the best .keras model.
        patience: EarlyStopping patience in epochs.
        phase_label: Short label for log file names.

    Returns:
        List of instantiated Keras callbacks.
    """
    import tensorflow as tf

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = PROJECT_ROOT / "reports" / f"training_log_{phase_label}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(log_path),
            append=False,  # start fresh for each run
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# History saving
# ─────────────────────────────────────────────────────────────────

def save_history(history, output_dir: Path, phase_label: str) -> None:
    """
    Save the Keras training history to a JSON file.

    The history contains per-epoch train/val loss and accuracy.
    These are the genuine values from the completed training run —
    never fabricated or estimated.

    Args:
        history: The History object returned by model.fit().
        output_dir: Directory to save the JSON.
        phase_label: Label for the output filename.
    """
    history_dict = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    history_dict["epochs_completed"] = len(history.epoch)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"training_history_{phase_label}.json"
    with open(out_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"  Training history saved: {out_path}")


# ─────────────────────────────────────────────────────────────────
# Validation evaluation
# ─────────────────────────────────────────────────────────────────

def evaluate_on_validation(
    model,
    val_dataset,
    val_labels: list[int],
    class_names: dict[int, str],
    output_dir: Path,
    phase_label: str,
) -> dict:
    """
    Evaluate the trained model on the VALIDATION set only.

    Computes and saves:
        - Overall accuracy
        - Macro-averaged F1, precision, recall
        - Per-class precision, recall, F1 support
        - Confusion matrix (PNG)
        - Full metrics (JSON)

    IMPORTANT: This function uses ONLY the validation set.
    The test manifest (data/splits/test_manifest.csv) is NOT
    used here. It is reserved for Phase 7 final evaluation.

    Args:
        model: The trained Keras model.
        val_dataset: tf.data.Dataset for the validation split.
        val_labels: Ground-truth integer labels (same order as dataset).
        class_names: Mapping of class index → display name string.
        output_dir: Directory to write metrics and confusion matrix.
        phase_label: Label for output filenames.

    Returns:
        Metrics dict (also written to disk).
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(f"\n  Evaluating on validation set ({phase_label})...")
    t0 = time.time()

    y_pred_probs = model.predict(val_dataset, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(val_labels)

    elapsed = time.time() - t0
    n_samples = len(y_true)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    display_names = [class_names.get(i, str(i)) for i in range(len(class_names))]
    report = classification_report(
        y_true, y_pred,
        target_names=display_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"  Samples evaluated   : {n_samples}")
    print(f"  Inference time      : {elapsed:.1f}s  "
          f"({1000 * elapsed / n_samples:.1f}ms per image)")
    print(f"  Validation accuracy : {acc:.4f}")
    print(f"  Macro F1            : {macro_f1:.4f}")

    # ── Save metrics JSON ─────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "phase": phase_label,
        "split": "validation",
        "n_samples": n_samples,
        "val_accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "inference_time_sec": round(elapsed, 3),
        "ms_per_image": round(1000 * elapsed / n_samples, 2),
        "classification_report": report,
        "note": (
            "Evaluated on validation set only. "
            "Test evaluation is reserved for Phase 7."
        ),
    }
    metrics_path = output_dir / f"val_metrics_{phase_label}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved       : {metrics_path}")

    # ── Save confusion matrix plot ────────────────────────────────
    fig_path = output_dir / f"confusion_matrix_{phase_label}.png"
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax,
        linewidths=0.3,
        linecolor="#DAD7CD",
    )
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(
        f"Aura Farming — Confusion Matrix ({phase_label}, validation set)",
        fontsize=12,
        pad=14,
    )
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrix    : {fig_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────
# Config update helper
# ─────────────────────────────────────────────────────────────────

def update_config_weights_path(weights_path: Path) -> None:
    """
    Write the trained model's path into config.yaml.

    After training, config.yaml's model.weights_path is updated
    so the Streamlit app and src/predictor.py can find the model
    without hard-coding any path.

    Only the weights_path line is replaced — no other config
    values are touched.

    Args:
        weights_path: Absolute path to the saved .keras file.
    """
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    rel = str(weights_path.relative_to(PROJECT_ROOT))

    with open(config_path) as f:
        text = f.read()

    if "weights_path: null" in text:
        text = text.replace("weights_path: null", f'weights_path: "{rel}"')
    else:
        # Replace an existing non-null value
        import re
        text = re.sub(
            r'weights_path:.*',
            f'weights_path: "{rel}"',
            text,
        )

    with open(config_path, "w") as f:
        f.write(text)

    print(f"\n  ✓  config.yaml updated: weights_path = {rel}")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Aura Farming — MobileNetV3 Small Baseline Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Feature extraction only (default):
  python training/train_baseline.py

  # Feature extraction + fine-tuning:
  python training/train_baseline.py --fine-tune

  # Custom batch size and epoch limit:
  python training/train_baseline.py --batch-size 16 --epochs 20

  # Read all settings from config.yaml (override nothing):
  python training/train_baseline.py --from-config
        """,
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Max epochs for Phase A (default from config.yaml)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (default from config.yaml)",
    )
    p.add_argument(
        "--fine-tune", action="store_true",
        help="Run Phase B fine-tuning after Phase A",
    )
    p.add_argument(
        "--fine-tune-at", type=int, default=None,
        help="Unfreeze base layers from this index onward (default from config.yaml)",
    )
    p.add_argument(
        "--fine-tune-epochs", type=int, default=None,
        help="Max epochs for Phase B (default from config.yaml)",
    )
    p.add_argument(
        "--dropout", type=float, default=None,
        help="Dropout rate in classification head (default from config.yaml)",
    )
    p.add_argument(
        "--no-augment", action="store_true",
        help="Disable training augmentation",
    )
    p.add_argument(
        "--from-config", action="store_true",
        help="Use all training settings from config.yaml (ignore other CLI args)",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config()
    ds_cfg  = cfg["dataset"]
    mdl_cfg = cfg["model"]
    trn_cfg = cfg.get("training", {})
    mapping = load_class_mapping()

    seed = ds_cfg["random_seed"]
    set_seeds(seed)

    # Import TensorFlow now (after seeds, before any tf.* calls)
    tf = _import_tensorflow()
    tf.random.set_seed(seed)

    # ── Resolve training hyperparameters ─────────────────────────
    # CLI values override config.yaml; config.yaml overrides defaults.
    batch_size    = args.batch_size    or trn_cfg.get("batch_size", 32)
    epochs_a      = args.epochs        or trn_cfg.get("phase_a_epochs", 30)
    patience_a    = trn_cfg.get("phase_a_patience", 5)
    epochs_b      = args.fine_tune_epochs or trn_cfg.get("phase_b_epochs", 20)
    patience_b    = trn_cfg.get("phase_b_patience", 7)
    fine_tune_at  = args.fine_tune_at  or trn_cfg.get("fine_tune_at", 100)
    dropout       = args.dropout       or trn_cfg.get("dropout_rate", 0.3)
    augment       = not args.no_augment and trn_cfg.get("augment", True)

    input_size  = tuple(mdl_cfg["input_size"])   # (224, 224)
    num_classes = len(mapping["classes"])
    class_names = {int(k): v["display_name"]
                   for k, v in mapping["classes"].items()}

    splits_dir  = PROJECT_ROOT / ds_cfg["splits_dir"]
    models_dir  = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"

    ckpt_a = PROJECT_ROOT / trn_cfg.get(
        "checkpoint_phase_a", "models/mobilenetv3_phase_a.keras"
    )
    ckpt_b = PROJECT_ROOT / trn_cfg.get(
        "checkpoint_phase_b", "models/mobilenetv3_baseline.keras"
    )

    print(f"\n{'='*60}")
    print(f"  Aura Farming — MobileNetV3 Small Baseline Training")
    print(f"  TensorFlow      : {tf.__version__}")
    print(f"  Input size      : {input_size}")
    print(f"  Classes         : {num_classes}")
    print(f"  Batch size      : {batch_size}")
    print(f"  Phase A epochs  : {epochs_a}  (patience={patience_a})")
    print(f"  Fine-tune       : {args.fine_tune}")
    if args.fine_tune:
        print(f"  Phase B epochs  : {epochs_b}  (patience={patience_b})")
        print(f"  Fine-tune at    : layer {fine_tune_at}")
    print(f"  Dropout         : {dropout}")
    print(f"  Augmentation    : {augment}")
    print(f"  Seed            : {seed}")
    print(f"{'='*60}")

    # ── Load manifests ────────────────────────────────────────────
    print("\n  Loading manifests...")
    train_manifest = splits_dir / "train_manifest.csv"
    val_manifest   = splits_dir / "val_manifest.csv"
    test_manifest  = splits_dir / "test_manifest.csv"

    # Fail loudly if any manifest is missing
    for p in [train_manifest, val_manifest]:
        if not p.exists():
            print(f"\n❌  Manifest not found: {p}")
            print("    Run training/prepare_dataset.py first.\n")
            sys.exit(1)

    if not test_manifest.exists():
        print(f"\n⚠  Test manifest not found: {test_manifest}")
        print("   (Expected. Test evaluation happens in Phase 7.)")

    train_paths, train_labels = load_split_paths_and_labels(train_manifest)
    val_paths,   val_labels   = load_split_paths_and_labels(val_manifest)

    print(f"  Training images   : {len(train_paths)}")
    print(f"  Validation images : {len(val_paths)}")
    print(f"  Test images       : NOT loaded (reserved for Phase 7)")

    # ── Class weights from TRAINING set only ──────────────────────
    class_weights = compute_class_weights(train_labels, num_classes)

    # ── tf.data pipelines ─────────────────────────────────────────
    print("  Building data pipelines...")
    train_ds = build_tf_dataset(
        train_paths, train_labels, input_size,
        batch_size=batch_size, augment=augment,
        shuffle=True, seed=seed,
    )
    val_ds = build_tf_dataset(
        val_paths, val_labels, input_size,
        batch_size=batch_size, augment=False,
        shuffle=False, seed=seed,
    )

    # ─────────────────────────────────────────────────────────────
    # Phase A — Feature extraction
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  Phase A: Feature Extraction (base layers frozen)")
    print(f"{'─'*60}")

    model = build_model(num_classes, input_size, dropout)
    model.summary(print_fn=lambda x: print(f"  {x}"))

    callbacks_a = build_callbacks(ckpt_a, patience_a, "phase_a")

    history_a = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_a,
        callbacks=callbacks_a,
        class_weight=class_weights,
        verbose=1,
    )

    print(f"\n  ✓  Phase A complete.")
    print(f"     Epochs run     : {len(history_a.epoch)}")
    print(f"     Best model     : {ckpt_a}")

    save_history(history_a, reports_dir, "phase_a")

    metrics_a = evaluate_on_validation(
        model, val_ds, val_labels, class_names,
        output_dir=reports_dir, phase_label="phase_a",
    )

    # ─────────────────────────────────────────────────────────────
    # Phase B — Fine-tuning (optional)
    # ─────────────────────────────────────────────────────────────
    if args.fine_tune:
        print(f"\n{'─'*60}")
        print(f"  Phase B: Fine-Tuning (unfreezing from layer {fine_tune_at})")
        print(f"{'─'*60}")

        # Reload the best Phase A checkpoint — not the in-memory model.
        # EarlyStopping with restore_best_weights=True should restore the
        # best epoch's weights in memory, but loading from the checkpoint
        # file guarantees we start Phase B from the verified best model
        # saved to disk. This is more robust against edge cases where
        # the in-memory weights may differ from the saved checkpoint
        # (e.g. if the callback and the save were not perfectly in sync).
        if not ckpt_a.exists():
            print(f"\n❌  Phase A checkpoint not found: {ckpt_a}")
            print("    Phase A must complete successfully before Phase B.")
            sys.exit(1)
        print(f"  Reloading best Phase A checkpoint: {ckpt_a}")
        model = tf.keras.models.load_model(str(ckpt_a))

        unfreeze_for_fine_tuning(
            model, fine_tune_at,
            learning_rate=trn_cfg.get("phase_b_lr", 1e-5),
        )

        callbacks_b = build_callbacks(ckpt_b, patience_b, "phase_b")

        history_b = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_b,
            callbacks=callbacks_b,
            class_weight=class_weights,
            verbose=1,
        )

        print(f"\n  ✓  Phase B complete.")
        print(f"     Epochs run     : {len(history_b.epoch)}")
        print(f"     Best model     : {ckpt_b}")

        save_history(history_b, reports_dir, "phase_b")

        metrics_b = evaluate_on_validation(
            model, val_ds, val_labels, class_names,
            output_dir=reports_dir, phase_label="phase_b",
        )
        final_model_path = ckpt_b
    else:
        final_model_path = ckpt_a

    # ── Update config.yaml ────────────────────────────────────────
    update_config_weights_path(final_model_path)

    print(f"\n{'='*60}")
    print(f"  Training complete.")
    print(f"  Saved model     : {final_model_path}")
    print(f"  Metrics         : reports/val_metrics_*.json")
    print(f"  Training logs   : reports/training_log_*.csv")
    print(f"\n  ⚠  The test manifest (data/splits/test_manifest.csv)")
    print(f"     has NOT been used. Reserve it for Phase 7 evaluation.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
