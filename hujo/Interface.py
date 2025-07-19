"""
    The plan here is to create a class that specifically facilitates interaction with an LLM.
    The LLM i'll design it for interfacing with will be my GPT-esque model.

    Handle:
        - Loading LLM weights,
        - Saving LLM weights,
        - Simple Training Methods?,
        - User Interaction (Chatting?)
        - Perhaps some way of measurement, but that may be part of training.
"""

import torch


class Interface:

    def __init__(self, model_class: type, model_config: dict, tokenizer: type):
        self.model = model_class(model_config)
        self.tokenizer = tokenizer

        self.config = model_config
        self.vocab_size:int = model_config.get('vocab_size')
        self.context_size: int = model_config.get('context_size')
        self.emb_dim: int = model_config.get('emb_dim')
        self.n_heads:int = model_config.get('n_heads')
        self.n_layers:int = model_config.get('n_layers')
        self.drop_rate:float = model_config.get('drop_rate')
        self.qkv_bias:bool = model_config.get('qkv_bias')

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"

    def load_model(self, file_path:str):
        self.model.load_state_dict(torch.load(file_path, map_location=self.device))
        print(f"Model Loaded From: {file_path}")

    def save_model(self, file_path:str):
        # TODO: When Training is implemented ensure I'm also saving/loading optimizer state
        file_path = f"{file_path}.pth"
        torch.save(self.model.state_dict(), file_path)
        print(f"Model State Saved to: {file_path}")

    def chat(self, max_new_tokens:int=10, ):
        self.model.eval()
        while usr_input := str(input(":> ")):
            response = self.generate_text_advanced(
                usr_input,
                max_new_tokens=max_new_tokens,
                temp=1.5,
                top_k=15
            )
            print(response)
            print('-' * 40)

    def text_to_tokens(self, text: str):
        encoded = self.tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        return torch.tensor(encoded).unsqueeze(0)

    def tokens_to_text(self, tokens: torch.Tensor):
        flat = tokens.squeeze(0)
        return self.tokenizer.decode(flat.tolist())

    def generate_text_advanced(self, text:str, max_new_tokens:int, temp:float=0.0, top_k:int=None, eos_id:str=None):
        """
            This function adds temperature scaling and top-k sampling into the text generation. This looks like it also
            includes the multinomial text generation methods as well
        """

        tokens = self.text_to_tokens(text)

        for _ in range(max_new_tokens):
            tokens_crop = tokens[:, -(self.context_size):]
            with torch.no_grad():
                logits: torch.Tensor = self.model(tokens_crop)
            logits = logits[:, -1, :]

            if top_k:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    condition=logits < min_val,
                    input=torch.tensor(float('-inf')).to(logits.device),
                    other=logits
                )
            if temp > 0.0:
                logits = logits / temp
                probabilities = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1)
            else:
                next_tokens = torch.argmax(logits, dim=-1, keepdim=True)

            # End Of Sequence ID
            if next_tokens == eos_id:
                break

            tokens = torch.cat((tokens, next_tokens), dim=1)
        return self.tokens_to_text(tokens)
