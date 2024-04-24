from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch

def load_model(model_path, tokenizer_path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, padding_side='left')
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.resize_token_embeddings(len(tokenizer))  # Ensure the model's embeddings match the tokenizer's vocabulary size
    return tokenizer, model

def chat(tokenizer, model):
    model.eval()  # Set the model to evaluation mode
    print("DuneGPT: Hello! What do you want to know about Dune?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Frame the input as a question and prompt for an answer
        prompt = f"Answer this Question: {user_input} Answer: "
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')
        chat_history_ids = new_user_input_ids  # Start fresh for each question

        chat_history_ids = model.generate(
            chat_history_ids,
            max_new_tokens=50,  # Limits the number of new tokens generated
            pad_token_id=tokenizer.pad_token_id,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=100,
            top_p=0.7,
            temperature=0.8
        )

        response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("DuneGPT:", response)

if __name__ == '__main__':
    tokenizer, model = load_model(model_path="../../models/Dune_model", tokenizer_path="../../models/tokenizer")
    chat(tokenizer, model)

