.PHONY: clean data tokenizer apply_tokenizer train inference all

# Define file paths
RAW_DATA_DIR := data/raw
PROCESSED_DATA_DIR := data/processed
TOKENIZER_DIR := models/tokenizer
MODEL_DIR := models/Dune_model
MODEL_FILE := $(MODEL_DIR)/pytorch_model.bin

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Make Dataset (Merge Raw Files)
data: $(PROCESSED_DATA_DIR)

$(PROCESSED_DATA_DIR):
	@mkdir -p $(PROCESSED_DATA_DIR)
	python src/data/make_dataset.py $(RAW_DATA_DIR) $(PROCESSED_DATA_DIR)

## Create Tokenizer
tokenizer: $(TOKENIZER_DIR)

$(TOKENIZER_DIR):
	@mkdir -p $(TOKENIZER_DIR)
	python src/models/create_tokenizer.py $(PROCESSED_DATA_DIR) $(TOKENIZER_DIR)

## Apply Tokenizer
apply_tokenizer: $(TOKENIZER_DIR) $(PROCESSED_DATA_DIR)
	python src/models/tokenize_corpus.py $(PROCESSED_DATA_DIR) $(TOKENIZER_DIR) $(PROCESSED_DATA_DIR)

train: $(MODEL_FILE)
	@mkdir -p $(MODEL_DIR)
	python src/models/model_trainer.py $(PROCESSED_DATA_DIR) $(MODEL_DIR)

inference: $(MODEL_FILE)
	python src/models/dune_chat.py $(MODEL_DIR)


## Clean up
clean:
	rm -rf $(PROCESSED_DATA_DIR) $(TOKENIZER_DIR) $(MODEL_DIR)
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
