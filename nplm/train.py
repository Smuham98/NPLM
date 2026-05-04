import torch
import yaml
import os
from torch.utils.data import DataLoader
from nplm.model import NPLM
from nplm.dataset import TextDataset
from nplm.utils import load_vocab
from nplm.optimizer import get_optimizer

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def train_model(config, data_loader, model, optimizer):
    """Train the NPLM model."""
    model.train()
    for step, batch in enumerate(data_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(config['device'])
        target_ids = batch['target_ids'].to(config['device'])

        outputs = model(input_ids, labels=target_ids)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        if step % config['eval_every'] == 0:
            print(f"Step {step}/{len(data_loader)}, Loss: {loss.item()}")

        if step >= config['max_steps']:
            break

def main():
    # Load configuration
    config = load_config('configs/tiny.yaml')
    
    # Load vocabulary
    vocab = load_vocab(config['tokenizer_path'])

    # Prepare dataset and dataloader
    train_dataset = TextDataset(config['data_dir'], vocab, context_size=config['context_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Initialize model and optimizer
    model = NPLM(config['vocab_size'], config['embedding_dim'], config['hidden_dim'], config['context_size'], config['dropout'])
    model.to(config['device'])
    optimizer = get_optimizer(model, config['lr'])

    # Training loop
    train_model(config, train_dataloader, model, optimizer)

    # Save the model
    os.makedirs(config['save_dir'], exist_ok=True)
    torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best.pt'))
    print("Model saved.")

if __name__ == "__main__":
    main()