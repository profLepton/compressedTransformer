from transformers import AutoTokenizer
import torch

# Load data

with open("tiny_shakespeare.txt", "r") as f:
    text = f.read()


# Tokenize data

batch_size = 1000






def batch_iterator():
    yield text


# Load tokenizer if it's there

try:
    tokenizer = AutoTokenizer.from_pretrained("tiny_shakespeare_tokenizer")
except:


    tokenizer = AutoTokenizer.from_pretrained("gpt2")


    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=75000, num_workers=4)



    new_tokenizer.save_pretrained("tiny_shakespeare_tokenizer")


    tokenizer = new_tokenizer


# Tokenize data

tokenized_text = tokenizer.encode(text)

# Create training pairs

train_pairs = []

input_size = 128

for i in range(0, len(tokenized_text) - input_size, input_size):
    train_pairs.append(
        (
            torch.tensor(tokenized_text[i : i + input_size], dtype=torch.int32),
            torch.tensor(tokenized_text[i + 1 : i + input_size + 1], dtype=torch.int32),
        )
    )


# Save training pairs


torch.save(train_pairs, "train_pairs.pt")


