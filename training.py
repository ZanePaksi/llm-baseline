"""
    Starts Chapter 5 Section 2
"""
import tiktoken
import torch
from PIL.PdfParser import encode_text
from torch import Tensor

from architecture import GPT_CONFIG_124M
from dataset import create_dataloader_v1
from gpt_model import generate_text_simple, GPTModel, generate_text_mutlinomial
from text_generation import calc_loss_batch, calc_loss_loader, text_to_token_ids, token_ids_to_text


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss: Tensor = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iiter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iiter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iiter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.positional_embedding_lookup.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_mutlinomial(model=model, token_ids=encoded, max_new_tokens=50, context_size=context_size)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


def test_training():
    with open("the-verdict.txt", 'r', encoding='utf-8') as file:
        text_data = file.read()

    tokenizer = tiktoken.get_encoding("gpt2")
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


    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    decoding_strategies(model, tokenizer)


# 5.3
def decoding_strategies(model, tokenizer):

    model.to("cpu")
    model.eval()

    token_ids = generate_text_mutlinomial(
        model=model,
        token_ids=text_to_token_ids("Every effort moves you", tokenizer),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"]
    )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


test_training()