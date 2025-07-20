from hujo.Model import GPTModel
from hujo.Interface import Interface
from hujo.Training import Trainer
import tiktoken

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

    # interface = Interface(GPTModel, GPT_CONFIG_124M, tiktoken.get_encoding('gpt2'), file_path="model.pth")
    # interface.load_model("model.pth")
    # interface.chat()

    trainer = Trainer(GPTModel, GPT_CONFIG_124M, tiktoken.get_encoding('gpt2'), TRAINER_CONFIG)
    trainer.prep_and_train("the-verdict.txt")

main()
