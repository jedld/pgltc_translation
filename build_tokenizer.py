

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Initialize an empty BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Choose a pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Initialize a trainer with the desired vocabulary size
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

# Train the tokenizer
import argparse

parser = argparse.ArgumentParser(description='Train a tokenizer.')
parser.add_argument('--corpus', type=str, help='Path to the corpus file.')
parser.add_argument('--tokenizer', type=str, help='Path to save the tokenizer file.')
args = parser.parse_args()

corpus_path = args.corpus
tokenizer_path = args.tokenizer

files = [corpus_path]  # Replace with the path to your text file

tokenizer.train(files, trainer)

# Save the tokenizer
tokenizer.save(tokenizer_path)
