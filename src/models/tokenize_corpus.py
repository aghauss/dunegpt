from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset

def load_tokenizer(path):
    try:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file = path) 
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

def tokenize_function(example, tokenizer,max_length):
    try:
        tokenized_example = tokenizer(
            example["text"],
            truncation = True,
            padding = True,
            max_length = max_length
        )
        return {"input_ids": tokenized_example["input_ids"]}
    
    except Exception as e:
        raise Exception(f"An error occurred during tokenization: {e}")



def load_text_dataset(file_path , encoding):
    """Load the text dataset using the specified file path and encoding."""
    try:
        return load_dataset("text", encoding=encoding, data_files=[file_path])
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def main():
    sequence_length = 156
    dataset_path = "../../data/processed/dune_full_corpus.txt"
    encoding_dune = 'ISO-8859-1'
    tokenizer_path = "../../models/dune_tokenizer.json"
    # Load corpus
    dataset = load_text_dataset(dataset_path,encoding=encoding_dune)

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Tokenize the corpus
    tokenized_datasets = dataset.map(
        lambda example: tokenize_function(example, tokenizer, sequence_length),
        batched=True, remove_columns=["text"]
    )
    
    tokenized_datasets.save_to_disk("../../data/processed/tokenized_corpus")
    

    


if __name__ == "__main__":
    main()