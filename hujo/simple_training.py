from hujo import torch


def test_training(text_data: str, interface, train_ratio: float):

    total_char = len(text_data)
    split_tokens = int(train_ratio * total_char)

    train_data = text_data[:split_tokens]
    val_data = text_data[split_tokens:]

    train_loader = create_dataloader_v1(
        train_data,
        tokenizer=interface.tokenizer,
        batch_size=2,
        max_length=interface.context_size,
        stride=interface.context_size,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        val_data,
        tokenizer=interface.tokenizer,
        batch_size=2,
        max_length=interface.context_size,
        stride=interface.context_size,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )


    interface.model.to(interface.device)

    optimizer = torch.optim.AdamW(interface.model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        interface, train_loader, val_loader, optimizer, num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you"
    )

    decoding_strategies(interface.model, interface.tokenizer, interface.device, start_context="Every effort moves you")

    torch.save(interface.model.state_dict(), "model.pth")


def train_model_simple(interface, train_loader, val_loader, optimizer, num_epochs, eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        interface.model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss: torch.Tensor = calc_loss_batch(input_batch, target_batch, interface.model, interface.device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(interface.model, train_loader, val_loader, interface.device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                      )

        generate_and_print_sample(
            interface.model, interface.tokenizer, interface.device, start_context
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
        token_ids = generate_text_advanced(model=model, token_ids=encoded, max_new_tokens=50, context_size=context_size)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()


# 5.3
def decoding_strategies(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.positional_embedding_lookup.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_advanced(model=model, token_ids=encoded, max_new_tokens=50, context_size=context_size)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))

    print("Output text:\n", decoded_text)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def generate_text_advanced(model, token_ids, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
        This function adds temperature scaling and top-k sampling into the text generation. This looks like it also
        includes the multinomial text generation methods as well
    """
    for _ in range(max_new_tokens):
        token_ids_crop = token_ids[:, -context_size:]
        with torch.no_grad():
            logits: torch.Tensor = model(token_ids_crop)
        logits = logits[:, -1, :]

        if top_k:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                condition=logits < min_val,
                input=torch.tensor(float('-inf')).to(logits.device),
                other=logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1)
            next_token_ids = torch.multinomial(probabilities, num_samples=1)
        else:
            next_token_ids = torch.argmax(logits, dim=-1, keepdim=True)

        if next_token_ids == eos_id:
            break

        token_ids = torch.cat((token_ids, next_token_ids), dim=1)
    return token_ids


# def topk_sampling():
#     next_token_logits = torch.tensor(
#         [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
#     )
#     top_k = 3
#     top_logits, top_pos = torch.topk(next_token_logits, top_k)
#     print("Top logits:", top_logits)
#     print("Top positions:", top_pos)
#
#     new_logits = torch.where(
#         condition=next_token_logits < top_logits[-1],
#         input=torch.tensor(float('-inf')),
#         other=next_token_logits
#     )
#     print(new_logits)
#
#     topk_probas = torch.softmax(new_logits, dim=0)
#     print(topk_probas)


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


def create_dataloader_v1(
                text, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    dataset = GPTDatasetV1(text, tokenizer, max_length, stride)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


class GPTDatasetV1(torch.utils.data.Dataset):

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

