from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerFast
import torch
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load tokenized data
    dataset_path = os.path.join(base_dir, "../../data/processed/tokenized_corpus")
    datasets = load_from_disk(dataset_path)

    # Loading trained tokenizer
    tokenizer_path = os.path.join(base_dir, "../../models/tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    # Create a configuration for the GPT-2 model
    model_config = GPT2Config(
        vocab_size=tokenizer.vocab_size,  # Use tokenizer's vocab size
        pad_token_id=tokenizer.pad_token_id,  # Ensure padding token ID is set
        n_ctx=156,  # Context size
        n_positions=156,  # Number of positional embeddings
        n_embd=512,  # Embedding size
        n_head=8,  # Number of attention heads
        n_layer=12  # Number of layers
    )

    model = GPT2LMHeadModel(model_config)
    
    # Setting up data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_path = os.path.join(base_dir, "../../models")

    training_args = TrainingArguments(
        output_dir=output_path, 
        overwrite_output_dir=True,
        num_train_epochs=15,
        per_device_train_batch_size=75,
        logging_dir=os.path.join(base_dir, './logs'),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"], 
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save tokenizer and model
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

if __name__ == "__main__":
    main()
