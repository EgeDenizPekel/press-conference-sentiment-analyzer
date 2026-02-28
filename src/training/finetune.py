"""
Fine-tune cardiffnlp/twitter-roberta-base-sentiment on NBA sports press conference labels.

Logs hyperparameters and per-epoch metrics to MLflow.
Saves the best checkpoint to models/fine-tuned-sports-sentiment/.

Usage:
    mlflow ui          # optional: launch tracking UI
    python -m src.training.finetune
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.training.dataset import (
    ID2LABEL,
    LABEL2ID,
    MODEL_NAME,
    build_dataset,
    build_training_labels,
)

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "fine-tuned-sports-sentiment"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS = 5
LR = 2e-5
TRAIN_BATCH = 16
EVAL_BATCH = 32
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 2


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred: tuple) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = float(accuracy_score(labels, preds))
    f1 = float(f1_score(labels, preds, average="macro"))
    return {"accuracy": acc, "f1": f1}


# ---------------------------------------------------------------------------
# MLflow callback
# ---------------------------------------------------------------------------

class MLflowMetricsCallback(TrainerCallback):
    """Log per-epoch eval metrics and per-step train loss to the active MLflow run."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> None:
        if not logs:
            return
        # Training step logs contain "loss" but not "eval_loss"
        if "loss" in logs and "eval_loss" not in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=state.global_step)

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict | None = None,
        **kwargs,
    ) -> None:
        if not metrics:
            return
        epoch = round(state.epoch) if state.epoch else 0
        mlflow.log_metrics(
            {
                "val_accuracy": metrics.get("eval_accuracy", 0.0),
                "val_f1":       metrics.get("eval_f1", 0.0),
                "val_loss":     metrics.get("eval_loss", 0.0),
            },
            step=epoch,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Building dataset...")
    combined = build_training_labels()
    dataset = build_dataset(combined)

    train_size = len(dataset["train"])
    val_size = len(dataset["validation"])
    print(f"Train: {train_size}  Val: {val_size}")

    print(f"\nLoading base model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        per_device_eval_batch_size=EVAL_BATCH,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(MODELS_DIR / "logs"),
        logging_steps=50,
        report_to="none",  # manual MLflow logging via callback
        save_total_limit=2,
    )

    mlflow.set_experiment("sports-sentiment-finetuning")

    with mlflow.start_run(run_name="twitter-roberta-sports") as run:
        mlflow.log_params({
            "model":        MODEL_NAME,
            "epochs":       EPOCHS,
            "lr":           LR,
            "batch_size":   TRAIN_BATCH,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "train_size":   train_size,
            "val_size":     val_size,
            "max_length":   256,
        })

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            compute_metrics=compute_metrics,
            callbacks=[
                MLflowMetricsCallback(),
                EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            ],
        )

        print("\nStarting fine-tuning...")
        trainer.train()

        # Final evaluation on best checkpoint
        eval_results = trainer.evaluate()
        print(f"\nFinal eval: {eval_results}")

        mlflow.log_metrics({
            "final_val_accuracy": eval_results.get("eval_accuracy", 0.0),
            "final_val_f1":       eval_results.get("eval_f1", 0.0),
            "final_val_loss":     eval_results.get("eval_loss", 0.0),
        })

        # Save model + tokenizer
        trainer.save_model(str(MODELS_DIR))
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(str(MODELS_DIR))
        print(f"\nModel saved to {MODELS_DIR}")

        # Log saved artifacts to MLflow
        mlflow.log_artifacts(str(MODELS_DIR), artifact_path="model")

        print(f"MLflow run ID: {run.info.run_id}")
        print(
            f"Final val accuracy: {eval_results.get('eval_accuracy', 0):.1%}  "
            f"F1: {eval_results.get('eval_f1', 0):.3f}"
        )


if __name__ == "__main__":
    main()
