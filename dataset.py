import torch
from torch.utils.data import Dataset, DataLoader


def create_dataloader_v1(
                text, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader



class GPTDatasetV1(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids: list = []
        self.target_ids: list = []

        # tokenizes the entire text
        token_ids = tokenizer.encode(text)

        # This divides the total tokens into a sliding window of overlapping sequences
        for i in range(0, len(token_ids) - max_length, stride):
            """
                input_chunk will be a slice of the total token_ids 
                Starting at i (being an incrementing index)
                Ending at i + max_length (how many tokens ahead we will include)
            """
            input_chunk = token_ids[i : i + max_length]
            """
                Target chunk will be the same idea as input_chunk
                but the starting and ending indexes of the slice will be shifted forward by 1
            """
            target_chunk = token_ids[i + 1 : i + max_length + 1]

            # Convert each list of tokens into pytorch tensors (multidimensional arrays)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]