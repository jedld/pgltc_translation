
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformer_model import TransformerModel
from transformers import BertTokenizer
from tokenizers import Tokenizer

# Load the pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load the trained tokenizer from a saved file
pg_tokenizer = Tokenizer.from_file('pg.json')

class TranslationDataset(Dataset):
    def __init__(self, data_file, source_vocab_file, target_vocab_file, split, target_pad_token='<pad>'):
        self.source_vocab = self.load_vocab(source_vocab_file)
        self.target_vocab = self.load_vocab(target_vocab_file)
        self.target_pad_idx = self.target_vocab[target_pad_token]  # Define padding index.
        self.data = []
        with open(data_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split('####')
                if len(parts) != 2:
                    raise ValueError(f"Error in line {i+1}: {line.strip()}")
                original, translated = parts
                self.data.append((original, translated))
        if split == 'train':
            self.data = self.data[:int(0.8*len(self.data))]
        elif split == 'test':
            self.data = self.data[int(0.8*len(self.data)):]

    def __getitem__(self, idx):
        original, translated = self.data[idx]
        original_indices = [self.source_vocab[token] for token in original.split()]
        translated_indices = [self.target_vocab[token] for token in translated.split()]
        return torch.tensor(original_indices), torch.tensor(translated_indices)

    @staticmethod
    def load_vocab(vocab_file):
        # Assuming each line in the vocab file represents a unique token
        vocab = {}
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for i, token in enumerate(f.readlines()):
                vocab[token.strip()] = i
        return vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        original, translated = batch
        original = original.to(device)
        translated = translated.to(device)
        optimizer.zero_grad()
        output = model(original, translated[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), translated[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            original, translated = batch
            original = original.to(device)
            translated = translated.to(device)
            output = model(original, translated[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), translated[:, 1:].reshape(-1))
            total_loss += loss.item()
            preds = output.argmax(dim=-1)
            total_correct += (preds == translated[:, 1:]).sum().item()
            total_tokens += translated[:, 1:].numel()
    accuracy = total_correct / total_tokens
    return total_loss / len(dataloader), accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--dim_feedforward', type=int, default=512, help='Dimension of feedforward network')
    parser.add_argument('--split', type=str, default='train', help='Split to use (train or test)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train = TranslationDataset(args.data_file, 'trans_vocab.txt', 'orig_vocab.txt', split='train')
    dataset_test = TranslationDataset(args.data_file, 'trans_vocab.txt', 'orig_vocab.txt', split='test')


    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(batch):
        originals, translateds = zip(*batch)
        
        # Assuming 'tokenizer' converts text strings into sequences of integers
        # Tokenize and convert to IDs
        originals_tokenized = [pg_tokenizer.encode(seq).ids for seq in originals]
        translateds_tokenized = [tokenizer.encode(seq, add_special_tokens=True, return_tensors="pt") for seq in translateds]
        
        originals_padded = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in originals_tokenized], 
                                        batch_first=True, 
                                        padding_value=0)
        translateds_padded = pad_sequence([seq.detach() for seq in translateds_tokenized], batch_first=True)

        return originals_padded, translateds_padded

    # Then, when creating your DataLoader, you pass the collate_fn:
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerModel(
        len(dataset_train.source_vocab), 
        len(dataset_train.target_vocab), 
        args.hidden_size, 
        args.num_heads, 
        args.num_layers,   # assuming this is the number of encoder layers
        args.num_layers,   # assuming you want the same number of decoder layers as encoder layers
        args.dim_feedforward, 
        args.dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=dataset_train.target_pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        loss_train = train(model, dataloader_train, criterion, optimizer, device)
        print(f'Epoch {epoch+1}/{args.num_epochs}, Train Loss: {loss_train:.4f}')
        if (epoch+1) % 10 == 0:
            loss_test, accuracy_test = test(model, dataloader_test, criterion, device)
            print(f'Test Loss: {loss_test:.4f}, Test Accuracy: {accuracy_test:.4f}')

if __name__ == '__main__':
    main()