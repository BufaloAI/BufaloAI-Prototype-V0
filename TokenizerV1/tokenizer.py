import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


class Tokenizer:

    def __init__(self, file_path, vocab_size = 50257, output_dim = 256, context_length = 1024):

        # self.file_path = file_path
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        self.raw_text = raw_text

        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.context_length = context_length

    def create_dataloader_v1(self, txt, batch_size, max_length, stride,
                            shuffle=True, drop_last=True, num_workers=0):
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        return dataloader

    def dataloader_with_sliding_window(self, batch_size = 8, max_length = 4):

        token_embedding_layer = torch.nn.Embedding(self.vocab_size, self.output_dim)
        pos_embedding_layer = torch.nn.Embedding(self.context_length, self.output_dim)


        dataloader = self.create_dataloader_v1(
            self.raw_text,
            batch_size=batch_size,
            max_length=max_length,
            stride=max_length
        )

        for batch in dataloader:
            x, y = batch

            token_embeddings = token_embedding_layer(x)
            pos_embeddings = pos_embedding_layer(torch.arange(max_length))

            input_embeddings = token_embeddings + pos_embeddings

            break

        return input_embeddings