import tiktoken
import torch
import torch.nn.modules.sparse
from torch.nn import Embedding, Dropout, Sequential, Linear

from architecture import LayerNorm, GPT_CONFIG_124M
from transformer import TransformerBlock


class GPTModel(torch.nn.Module):

    def __init__(self, config: dict):
        super().__init__()
        # Embeddings are lookup tables that store embeddings of a fixed dictionary size
        self.token_embedding_lookup: Embedding = torch.nn.Embedding(config['vocab_size'], config['emb_dim'])
        self.positional_embedding_lookup: Embedding = torch.nn.Embedding(config['context_length'], config['emb_dim'])
        self.dropout: Dropout = torch.nn.Dropout(config['drop_rate'])

        self.transformer_blocks: Sequential = torch.nn.Sequential(*[TransformerBlock(config) for _ in range(config['n_layers'])])
        self.final_norm: LayerNorm = LayerNorm(config['emb_dim'])
        self.out_head: Linear = torch.nn.Linear(config['emb_dim'], config['vocab_size'], bias=False)

    def forward(self, token_ids: torch.Tensor):
        batch_size, seq_len = token_ids.shape
        token_embeddings: torch.Tensor = self.token_embedding_lookup(token_ids)
        positional_embeddings: torch.Tensor = self.positional_embedding_lookup(torch.arange(seq_len, device=token_ids.device))

        hidden_embeddings: torch.Tensor = token_embeddings + positional_embeddings
        hidden_embeddings: torch.Tensor = self.dropout(hidden_embeddings)
        hidden_embeddings: torch.Tensor = self.transformer_blocks(hidden_embeddings)
        hidden_embeddings: torch.Tensor = self.final_norm(hidden_embeddings)
        logits: torch.Tensor = self.out_head(hidden_embeddings)

        return logits

def test_type():

    # lookup = torch.nn.Embedding(50276, 768)
    # token_ids = torch.tensor([1, 4, 6, 6])
    # print(lookup(token_ids))

    m = torch.nn.Linear(20, 30)
    input_nums = torch.randn(128, 20)
    output = m(input_nums)
    print(output)


def try_gpt_model_z():
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    model.eval()
    out = generate_text_simple(
        model=model,
        token_ids=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M['context_length']
    )

    print(out)
    print(f"{len(out[0])=}")

    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)

    # print("Input batch:\n", batch)
    # print("\nOutput shape:", out.shape)
    # print(out)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params:,}")
    # print("Token embedding layer shape:", model.token_embedding_lookup.weight.shape)
    # print("Output layer shape:", model.out_head.weight.shape)
    #
    # total_params_gpt2 = (
    #         total_params - sum(p.numel()
    #                            for p in model.out_head.parameters())
    # )
    # print(f"Number of trainable parameters "
    #       f"considering weight tying: {total_params_gpt2:,}"
    #       )
    #
    # total_size_bytes = total_params * 4
    # total_size_mb = total_size_bytes / (1024 * 1024)
    # print(f"Total size of the model: {total_size_mb:.2f} MB")


def generate_text_simple(model, token_ids, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        token_ids_crop = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(token_ids_crop)

        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        token_ids_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        token_ids = torch.cat((token_ids, token_ids_next), dim=1)

    return token_ids


def generate_text_mutlinomial(model, token_ids, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        token_ids_crop = token_ids[:, -context_size:]
        with torch.no_grad():
            logits = model(token_ids_crop)

        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probabilities, num_samples=1)
        token_ids = torch.cat((token_ids, next_token_id), dim=1)

    return token_ids


def multinomial_probas_example():
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }
    inverse_vocab = {v: k for k, v in vocab.items()}
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )
    probas = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas).item()
    print(inverse_vocab[next_token_id])

    torch.manual_seed(123)
    next_token_id = torch.multinomial(probas, num_samples=1).item()
    print(inverse_vocab[next_token_id])

    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item()
              for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


multinomial_probas_example()

# TODO: Important func here and above with multinomial
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits / temperature
    return torch.softmax(scaled_logits, dim=0)

def chat():
    tokenizer = tiktoken.get_encoding("gpt2")
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()

    while usr_input := str(input(":> ")):
        encoded = tokenizer.encode(usr_input)
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        out = generate_text_simple(
            model=model,
            token_ids=encoded_tensor,
            max_new_tokens=10,
            context_size=GPT_CONFIG_124M['context_length']
        )
        decoded_text = tokenizer.decode(out.squeeze(0).tolist())
        print(decoded_text)
        print('-' * 40)

