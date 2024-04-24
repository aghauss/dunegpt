# Dunegpt
Interactive GPT2-based Chatbot for Dune fans to ask questions about the franchise.

## Project Description

Dunegpt is a project that trains a GPT2-model from scratch utilizing the books of the Dune Franchise of Frank Herbert's Dune books. The project aims to enable fans to ask questions about the franchise, ask for stories in the style of Dune and overall enhance the dune experience.

## Installation

1. Clone the repository:
   ```zsh
   git clone https://github.com/aghauss/dunegpt.git
2. Navigate to the project directory:
   ```zsh
   cd dunegpt
3. Create a virtual enviroment and activate it (for macOS and Linux)
    ```zsh
    python -m venv dunegpt-env
    source venv/bin/activate
4. Install requirements:
   ```zsh
   pip install -r requirements.txt
5. Make corpus for training available:
     - Move the corpus you want to use (e.g. the Dune series) to the data/raw folder. Each book should be available as .txt file.
     - The corpus will be automatically merged by a script.


## Usage

Once the environment is set up and the corpus is available, you can use the provided Makefile to execute the workflow. Here are the main targets:


1. **Make Dataset (Merge Raw Files)**:
   - This target merges the raw files in the `data/raw` directory to create the dataset.
     ```zsh
     make data
     ```

2. **Create Tokenizer**:
   - This target creates the tokenizer based on the corpus.
     ```zsh
     make tokenizer
     ```

3. **Apply Tokenizer**:
   - This target applies the tokenizer to the full corpus.
     ```zsh
     make apply_tokenizer
     ```

4. **Train Model**:
   - This target trains the GPT2-model from scratch using the corpus. Sufficient GPU memory should be available to perfomr this step!
     ```zsh
     make train
     ```

5. **Inference**:
   - This target runs the chatbot using the trained model.
     ```zsh
     make inference
     ```

6. **Clean**:
   - This target cleans up the processed data, tokenizer, and trained model files.
     ```zsh
     make clean
     ```

These commands can be executed in your terminal by running `make <target>`, where `<target>` is one of the targets listed above. Ensure you have the necessary data files in the `data/raw` directory before running the `make data` target.




Project Organization
------------


    ├── data               <- Data directory 
    │   ├── processed      <- Contains the corpus as one .txt file and the fully tokenized corpus 
    │   └── raw            <- Corpus needs to be placed here.
    │
    ├── models             <- Contains the trained tokenizer and the GPT2 model after training.
    │
    │
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to merge and harmonize several txt files (the individual books) 
    │   │   └── make_dataset.py
    │   │
    │   ├── models         <- Scripts to train tokenizer and GPT2 model.
    │   │   │                 predictions
    │   │   ├── create_tokenizer.py
    │   │   └── dune_chat.py
    │   │   └── model_trainer.py
    │   │   └── tokenize_corpus.py
    │   │   └── tokenizer_config.json
    ├── Makefile           <- Makefile to replicate entire workflow.
    ├── README.md          <- The file you are currently reading.
    ├── requirements.txt   <- The requirements file for reproducing the environment


## Dependencies

- [accelerate](https://pypi.org/project/accelerate/) (version 0.29.3)
  - License: MIT License

- [ipykernel](https://pypi.org/project/ipykernel/) (version 6.29.4)
  - License: BSD 3-Clause License

- [ipython](https://pypi.org/project/ipython/) (version 8.23.0)
  - License: BSD 3-Clause License

- [pandas](https://pypi.org/project/pandas/) (version 2.2.2)
  - License: BSD 3-Clause License

- [tensorflow](https://pypi.org/project/tensorflow/) (version 2.16.1)
  - License: Apache License 2.0

- [tokenizers](https://pypi.org/project/tokenizers/) (version 0.19.1)
  - License: MIT License

- [torch](https://pypi.org/project/torch/) (version 2.2.2)
  - License: BSD 3-Clause License

- [transformers](https://pypi.org/project/transformers/) (version 4.40.0)
  - License: Apache License 2.0


## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
