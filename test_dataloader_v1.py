from dataset import create_dataloader_v1
import tiktoken

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

tokenizer = tiktoken.get_encoding("gpt2")
data_loader = create_dataloader_v1(raw_text, tokenizer, batch_size=1, max_length=4, stride=1, shuffle=False)

data_iter = iter(data_loader)

first_batch = next(data_iter)
print(first_batch)