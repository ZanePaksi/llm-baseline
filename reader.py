from tokenizer import SimpleTokenizerV1, SimpleTokenizerV2
from dataset import create_dataloader_v1
import embedding
import tiktoken
import re

"""
    Tiktoken docs: https://pypi.org/project/tiktoken/
"""


def token_embedding_with_abs_pos():
    raw_text = load_text()
    tokenizer = tiktoken.get_encoding("gpt2")

    dataloader = create_dataloader_v1(raw_text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"Token IDS:\n{inputs}")
    print(f"Inputs Shape:\n{inputs.shape}\n")

    token_embedding_layer = embedding.get_bpe_embedding_layer(vocab_size=50257, output_dim=256)
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape, '\n')
    pos_embeddings = embedding.get_abs_embedding_layer(context_length=4, output_dim=256)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)



def first_token_embedding():
    raw_text = load_text()
    tokenizer = tiktoken.get_encoding("gpt2")

    dataloader = create_dataloader_v1(raw_text, tokenizer, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"Inputs:\n{inputs}")
    print(f"Targets:\n{targets}")


def sliding_window_demo(raw_text, tokenizer):
    encoded_text = tokenizer.encode(raw_text)
    print(len(encoded_text))

    enc_sample = encoded_text[50:]
    context_size = 4
    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]
    print(f"x: {x}")
    print(f"y:     {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


def tokenizers_first_pass():
    raw_text = load_text()

    tokens: list = tokenize(raw_text)
    vocab: dict = build_vocab(tokens)
    print(f"VOCAB LENGTH: {len(vocab)}")

    v1 = SimpleTokenizerV1(vocab)
    v2 = SimpleTokenizerV2(vocab)

    tik = tiktoken.get_encoding("gpt2")

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of someunknownPlace."
    text = " <|endoftext|> ".join((text1, text2))

    other_text = "Akwirw ier"

    # test_tokenizer(v2, text)
    test_bpe_tokenizer(tik, other_text)


def load_text() -> str:
    raw_text: str

    with open("the-verdict.txt", "r", encoding="utf-8") as file:
        raw_text = file.read()

    return raw_text


def tokenize(text: str) -> list:
    preproc = []
    preproc = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    preproc = [item.strip() for item in preproc if item.strip()]
    return preproc


def build_vocab(tokens: list) -> dict:
    all_tokens = sorted(set(tokens))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    # Dictionary comprehension to represent ordered list as key:value
    vocab = {token:integer for integer, token in enumerate(all_tokens)}
    return vocab


def test_tokenizer(tokenizer, text):
    print(f"INPUT:\n{text}\n")

    ids = tokenizer.encode(text)
    print(f"ENCODED:\n{ids}\n")

    decoded_text = tokenizer.decode(ids)
    print(f"DECODED:\n{decoded_text}\n")


def test_bpe_tokenizer(tokenizer, text):
    print(f"INPUT:\n{text}\n")

    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print(f"ENCODED:\n{ids}\n")

    decoded_text = tokenizer.decode(ids)
    print(f"DECODED:\n{decoded_text}\n")


if __name__ == "__main__":
    main()