"""
    Chapter 5
"""

import tiktoken
import torch
from gpt_model import GPTModel, generate_text_simple
from architecture import GPT_CONFIG_124M
from dataset import create_dataloader_v1


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

    """
        Perplexity is a measure often used alongside cross entropy loss to evaluate the performance of models in 
        tasks like language modeling. It can provide a more interpretable way to understand the uncertainty of a model 
        in predicting the next token in a sequence. 
        
        Perplexity can be calculated as 
            perplexity = torch.exp(loss)
             
        which returns tensor(48725.8203) when applied to the previously calculated loss. 
    """


def training_data_and_loss():
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPTModel(GPT_CONFIG_124M)
    text_data = ''
    with open("the-verdict.txt", 'r', encoding='utf-8') as file:
        text_data = file.read()

    total_char = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    # print(f"{total_char=}")
    # print(f"{total_tokens=}")

    train_ratio = 0.9
    split_tokens = int(train_ratio * total_char)
    train_data = text_data[:split_tokens]
    val_data = text_data[split_tokens:]

    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        tokenizer=tokenizer,
        batch_size=2,
        max_length=GPT_CONFIG_124M['context_length'],
        stride=GPT_CONFIG_124M['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # # Verifying the loaders were created correctly
    # print("Train loader:")
    # for x, y in train_loader:
    #     print(x.shape, y.shape)
    #
    # print("\nValidation loader:")
    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


