from dataset import create_dataloader_v1
from hujo.Model import GPTModel
from hujo import torch
from hujo.Interface import Interface
from hujo import utils
from hujo.Training import Trainer
from hujo import tiktoken
from hujo.simple_training import test_training

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # refers to a vocabulary of 50,257 words, as used by the BPE tokenizer
    "context_size": 256,     # maximum number of input tokens the model can handle via the positional embeddings
    "emb_dim": 768,          # represents the embedding size, transforming each token into a 768-dimensional vector.
    "n_heads": 12,           # indicates the count of attention heads in the multi-head attention mechanism
    "n_layers": 12,          # specifies the number of transformer blocks in the model
    "drop_rate": 0.1,        # indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting
    "qkv_bias": False        # determines whether to include a bias vector in the Linear layers of the multi-head attention
}


TRAINER_CONFIG = {
    "train_ratio": 0.9,      # The percentage of the data we will use to train (0.1 will be used for validation)
    "batch_size": 2,
    "num_epochs": 10,
    "eval_freq": 5,
    "eval_iter": 5,
    "start_context": "Every effort moves you"

}

def main():

    device = utils.get_torch_device()
    torch.cuda.empty_cache()

    interface = Interface(GPTModel, GPT_CONFIG_124M, tiktoken.get_encoding('gpt2'), device, "model.pth")
    interface.chat()

    # trainer = Trainer(GPTModel, GPT_CONFIG_124M, tiktoken.get_encoding('gpt2'), device, TRAINER_CONFIG)
    # trainer.prep_and_train("the-verdict.txt")

    # text_data = load_text_data("the-verdict.txt")
    # text_data = load_text_data("data/internet_archive_scifi_v3.txt")
    # test_training(text_data[:100_000], interface, 0.9)


def load_text_data(file_path):
    text_data = ""
    with open(file_path, 'r', encoding='utf-8') as file:
        text_data = file.read()
    return text_data


# def test_training(trainer):
#     with open("the-verdict.txt", 'r', encoding='utf-8') as file:
#         text_data = file.read()
#
#     total_char = len(text_data)
#
#     split_tokens = int(0.9 * total_char)
#     train_data = text_data[:split_tokens]
#     val_data = text_data[split_tokens:]
#
#     train_loader = create_dataloader_v1(
#         train_data,
#         tokenizer=trainer.tokenizer,
#         batch_size=2,
#         max_length=trainer.context_size,
#         stride=trainer.context_size,
#         drop_last=True,
#         shuffle=True,
#         num_workers=0
#     )
#
#     val_loader = create_dataloader_v1(
#         val_data,
#         tokenizer=trainer.tokenizer,
#         batch_size=2,
#         max_length=trainer.context_size,
#         stride=trainer.context_size,
#         drop_last=True,
#         shuffle=True,
#         num_workers=0
#     )
#
#     trainer.model.to(trainer.device)
#
#     optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.0004, weight_decay=0.1)
#
#     print(f"{'*' * 10} {optimizer.device}")
#
#     num_epochs = 10
#     train_losses, val_losses, tokens_seen = train_model_simple(
#         trainer.model, train_loader, val_loader, optimizer, trainer.device, num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#         start_context="Every effort moves you", tokenizer=trainer.tokenizer
#     )
#
#     decoding_strategies(trainer)
#
#     torch.save(trainer.model.state_dict(), "model.pth")
#
#
# def decoding_strategies(trainer):
#
#     trainer.model.to("cpu")
#     trainer.model.eval()
#
#     text = trainer.generate_text_advanced("Every effort moves you", 25, 0.0, 0)
#
#     print("Output text:\n", text)


main()
