import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

import torch
import numpy as np
from datasets import load_dataset
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import transformers
import inspect

# Select GPU (CUDA) if available, else fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {device}")

# ============================
# ✅ Model Checkpoint
# ============================
# Pre-trained base model to fine-tune with LoRA
model_checkpoint = "distilbert-base-uncased"

def print_model_size(path):
    """
    Calculate and print the total size of files in the given directory path,
    representing the disk footprint of a saved model.

    Args:
        path (str): Directory path containing model files.
    """
    """Calculate and print the size of the saved model files in MB."""
    size = 0
    for f in os.scandir(path):
        size += os.path.getsize(f)
    print(f"Model size: {(size / 1e6):.2f} MB")

def print_trainable_parameters(model, label):
    """
    Print the number of trainable vs total parameters in a model.

    Args:
        model (torch.nn.Module): The model to inspect.
        label (str): Label to indicate which model is being reported.
    """
    """Display total vs trainable parameters in the model."""
    parameters, trainable = 0, 0
    for _, p in model.named_parameters():
        parameters += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    print(f"{label} trainable parameters: {trainable:,}/{parameters:,} ({100 * trainable / parameters:.2f}%)")

def build_lora_model(num_labels):
    """
    Load a base model and configure it for LoRA (Low-Rank Adaptation) fine-tuning.

    Args:
        num_labels (int): Number of output classes.

    Returns:
        torch.nn.Module: A model prepared for LoRA fine-tuning.
    """
    """Load pre-trained base model and wrap it with LoRA configuration for parameter-efficient fine-tuning."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=num_labels,
    )
    print_trainable_parameters(model, label="Base model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )

    lora_model = get_peft_model(model, lora_config)
    print_trainable_parameters(lora_model, label="LoRA")
    lora_model.to(device)
    return lora_model

def preprocess_function(examples, tokenizer):
    """
    Tokenize and format text examples for input into a transformer model.

    Args:
        examples (dict): A batch of input examples from the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to process the input text.

    Returns:
        dict: Tokenized inputs with labels.
    """
    """Lowercase and tokenize input text; map labels for training."""
    texts = [str(text).lower().strip() for text in examples["text"]]
    result = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    result["labels"] = examples["labels"]
    return result

if __name__ == "__main__":
    print("\nStarting LoRA fine-tuning demo...")
    print("✅ Transformers version:", transformers.__version__)
    print("✅ TrainingArguments module:", TrainingArguments.__module__)
    print("✅ TrainingArguments file path:", inspect.getfile(TrainingArguments))
    print("✅ Python executable:", os.sys.executable)
    print(f"Using model: {model_checkpoint}")

    # ============================
    # ✅ Load and prepare datasets
    # ============================
    # Subset of IMDB and AG News for binary and multiclass classification
    dataset1 = load_dataset("imdb", split="train[:1000]")
    dataset2 = load_dataset("ag_news", split="train[:1000]")
    print(f"Dataset 1 size: {len(dataset1)} examples")
    print(f"Dataset 2 size: {len(dataset2)} examples")

    # Standardize label column
    dataset1 = dataset1.rename_column("label", "labels")
    dataset2 = dataset2.rename_column("label", "labels")

    # 80-20 train/test split
    train_size = int(0.8 * len(dataset1))
    dataset1_train = dataset1.select(range(train_size))
    dataset1_test = dataset1.select(range(train_size, len(dataset1)))
    dataset2_train = dataset2.select(range(train_size))
    dataset2_test = dataset2.select(range(train_size, len(dataset2)))

    # Initialize tokenizer and padding handler
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Task-specific training configuration
    config = {
        "sentiment": {
            "train_data": dataset1_train,
            "test_data": dataset1_test,
            "num_labels": 2,
            "epochs": 5,
            "path": "./lora-sentiment",
        },
        "topic": {
            "train_data": dataset2_train,
            "test_data": dataset2_test,
            "num_labels": 4,
            "epochs": 5,
            "path": "./lora-topic",
        },
    }

    # Apply preprocessing and format as PyTorch tensors
    print("Preprocessing datasets...")
    for cfg in config.values():
        cfg["train_data"] = cfg["train_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],
        )
        cfg["test_data"] = cfg["test_data"].map(
            lambda x: preprocess_function(x, tokenizer),
            batched=True,
            remove_columns=["text"],
        )
        cfg["train_data"].set_format("torch")
        cfg["test_data"].set_format("torch")

    # Trainer arguments: control training, evaluation, logging, saving
    training_arguments = TrainingArguments(
        output_dir="./checkpoints",
        learning_rate=2e-4,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        warmup_steps=100,
        seed=42,
    )

    # Compute accuracy metric during evaluation
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": accuracy}

    # Train LoRA models on both tasks
    for name, cfg in config.items():
        print(f"\nTraining {name} classifier...")
        model = build_lora_model(cfg["num_labels"])

        trainer = Trainer(
            model=model,
            args=training_arguments,
            train_dataset=cfg["train_data"],
            eval_dataset=cfg["test_data"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        eval_results = trainer.evaluate()
        print(f"Evaluation accuracy: {eval_results['eval_accuracy']:.4f}")
        trainer.save_model(cfg["path"])
        print_model_size(cfg["path"])

    # ============================
    # ✅ Inference / Prediction
    # ============================
    def predict_text(text, model_path, num_labels, task_type):
        """
        Perform prediction on a single text input using a fine-tuned model.

        Args:
            text (str): The input text to classify.
            model_path (str): Path to the fine-tuned model.
            num_labels (int): Number of classes.
            task_type (str): Task type, e.g., 'sentiment' or 'topic'.

        Returns:
            Tuple[str, float]: Predicted label and confidence score.
        """
        from peft import PeftModel  # Ensure PEFT model is used correctly

        # Load base model and apply LoRA adapter
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint, num_labels=num_labels
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        model.to(device)
        model.eval()

        # Tokenize and move inputs to appropriate device
        inputs = tokenizer(
            text.lower().strip(),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        ).to(device)

        # Run inference and extract probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted_class].item()

        # Map predicted class index to readable label
        label_map = {
            "sentiment": {0: "Negative", 1: "Positive"},
            "topic": {
                0: "World",
                1: "Sports",
                2: "Business",
                3: "Science/Technology",
            },
        }

        return label_map[task_type][predicted_class], confidence

    # Define example texts and expected outcomes
    test_texts = [
        {
            "text": "This movie was absolutely fantastic! The acting was superb.",
            "model": "sentiment",
            "num_labels": 2,
            "task_type": "sentiment",
            "expected": "Positive",
        },
        {
            "text": "The worst film I've ever seen. Complete waste of time.",
            "model": "sentiment",
            "num_labels": 2,
            "task_type": "sentiment",
            "expected": "Negative",
        },
        {
            "text": "Tesla stock surges 20% after strong quarterly earnings report.",
            "model": "topic",
            "num_labels": 4,
            "task_type": "topic",
            "expected": "Business",
        },
        {
            "text": "New AI model achieves breakthrough in protein folding.",
            "model": "topic",
            "num_labels": 4,
            "task_type": "topic",
            "expected": "Science/Technology",
        },
    ]

    print("\nRunning predictions on test examples:")
    for test in test_texts:
        prediction, confidence = predict_text(
            test["text"],
            config[test["model"]]["path"],
            test["num_labels"],
            test["task_type"],
        )
        print(f"\nText: {test['text']}")
        print(f"Expected: {test['expected']}")
        print(f"Predicted: {prediction}")
        print(f"Confidence: {confidence:.2%}")
