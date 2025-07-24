from hujo.Interface import Interface
from hujo import tiktoken
from hujo import torch
from hujo import utils

class Trainer(Interface):

    def __init__(self, model_class: type, model_config: dict, tokenizer: tiktoken.Encoding, device, trainer_config: dict, file_path=''):
        super().__init__(model_class, model_config, tokenizer, device)

        self.train_ratio = trainer_config.get('train_ratio')
        self.batch_size = trainer_config.get('batch_size')
        self.num_epochs = trainer_config.get('num_epochs')
        self.eval_freq = trainer_config.get('eval_freq')
        self.eval_iter = trainer_config.get('eval_iter')
        self.start_context = trainer_config.get('start_context')

        self.optimizer = None

        print(self.model.device)

    def calc_loss_loader(self, data_loader, num_batches=None):
        total_loss = 0.
        if len(data_loader) == 0:
            return float('nan')
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))

        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = self.calc_batch_loss(input_batch, target_batch)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches

    def evaluate_model(self, train_loader, val_loader):
        self.model.eval()
        with torch.no_grad():
            train_loss = self.calc_loss_loader(train_loader, num_batches=self.eval_iter)
            val_loss = self.calc_loss_loader(val_loader,num_batches=self.eval_iter)
        self.model.train()
        return train_loss, val_loss

    def calc_batch_loss(self, input_batch, target_batch):
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        logits = self.model(input_batch)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
        return loss

    def train_model_simple(self, train_loader, val_loader):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        for epoch in range(self.num_epochs):
            self.model.train()

            for input_batch, target_batch in train_loader:
                self.optimizer.zero_grad()
                loss: torch.Tensor = self.calc_batch_loss(input_batch, target_batch)
                loss.backward()
                self.optimizer.step()
                tokens_seen += input_batch.numel()
                global_step += 1

                if global_step % self.eval_freq == 0:
                    train_loss, val_loss = self.evaluate_model(train_loader, val_loader)
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                          f"Train loss {train_loss:.3f}, "
                          f"Val loss {val_loss:.3f}"
                          )

            self.generate_and_print_sample()

    def generate_and_print_sample(self):
        self.model.eval()
        text = self.generate_text_advanced(self.start_context, max_new_tokens=50)
        print(text.replace("\n", " "))
        self.model.train()

    def prep_and_train(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as data_file:
            text = data_file.read()

        total_char = len(text)
        split_tokens = int(self.train_ratio * total_char)

        train_data = text[:split_tokens]
        val_data = text[split_tokens:]

        train_loader = self.create_data_loader(train_data)
        val_loader = self.create_data_loader(val_data)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.0004, weight_decay=0.1)
        self.train_model_simple(train_loader, val_loader)

    def create_data_loader(self, text, shuffle=True, drop_last=True, num_workers=0):
        data_set = GPTDatasetV1(text, self.tokenizer, self.context_size, self.context_size // 2)

        data_loader = torch.utils.data.DataLoader(
            data_set,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )
        return data_loader

    def decoding_strategies(self):
        self.model.eval()

        text = self.generate_text_advanced("Every effort moves you", 25, 1.4, 20)

        print("Output text:\n", text)

# TODO: This can maybe get optimized. Need to explore training methodologies and dataset prep.
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