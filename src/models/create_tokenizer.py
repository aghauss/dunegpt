import json
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def load_config(file_path):
    with open(file_path, "r") as config_file:
        config = json.load(config_file)
    return config

def load_text_dataset(file_path , encoding):
    """Load the text dataset using the specified file path and encoding."""
    try:
        return load_dataset("text", encoding=encoding, data_files=[file_path])
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


class TextTokenizer:
    def __init__(self, vocab_size=5000, special_tokens=None):
        if special_tokens is None:
            special_tokens = ["[UNK]", "[PAD]"]

        self.tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

        self.trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)

        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def batch_iterator(self, dataset, batch_size=1000):
        """Generate batches of text."""
        for i in range(0, len(dataset["train"]), batch_size):
            yield dataset["train"][i:i + batch_size]["text"]

    def train(self, dataset):
        """Train the tokenizer using the provided dataset."""
        self.tokenizer.train_from_iterator(self.batch_iterator(dataset), trainer=self.trainer, length=len(dataset["train"]))

    def save(self, path="tokenizer.json"):
        """Save the trained tokenizer to a file."""
        self.tokenizer.save(path)

    def load_fast_tokenizer(self, path="tokenizer.json"):
        """Load a fast tokenizer."""
        self.fast_tokenizer = PreTrainedTokenizerFast(tokenizer_file=path)
        self.fast_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return self.fast_tokenizer

def main():
    
    # Load configuration
    config = load_config("tokenizer_config.json")
    # Load text dataset
    dataset_dict = load_text_dataset(file_path= config['dataset_config']['file_path'],encoding = config['dataset_config']['encoding'])

    # Initialize and train tokenizer
    tokenizer = TextTokenizer(vocab_size=config['tokenizer_config']['vocab_size'], special_tokens=config['tokenizer_config']['special_tokens'])
    tokenizer.train(dataset_dict)

    # Save tokenizer
    tokenizer.save(path=config['save_config']['tokenizer_path'])


if __name__ == "__main__":
    main()
