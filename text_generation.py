"""
    Chapter 5
"""

import tiktoken
import torch
from gpt_model import GPTModel, generate_text_simple, GPT_CONFIG_124M


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def test_text():
    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    token_ids = generate_text_simple(
        model=model,
        token_ids=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"]
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


def text_gen_loss():
    tokenizer = tiktoken.get_encoding("gpt2")

    inputs = torch.tensor([[16833, 3626, 6100],  # ["every effort moves",
                           [40, 1107, 588]])  # "I really like"]

    targets = torch.tensor([[3626, 6100, 345],  # [" effort moves you",
                            [1107, 588, 11311]])  # " really like chocolate"]

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(logits, dim=-1)
    print(probas.shape)

    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print(f"{token_ids=}")

    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    text_token = 0
    target_probas_1 = probas[text_token, [0, 1, 2], targets[text_token]]
    print("Text 1:", target_probas_1)

    text_token = 1
    target_probas_2 = probas[text_token, [0, 1, 2], targets[text_token]]
    print("Text 2:", target_probas_1)

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    # turnning the negative mean of probabilities to positive is known as 'cross entropy loss'
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    print('\n\n')
    # prepping to avoid all the loss calculations in replacement for the pytorch cross_entropy function
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)
    # see that loss is calculated the same as neg_avg_log_probas?

text_gen_loss()

