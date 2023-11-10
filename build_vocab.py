import argparse
import re
from collections import Counter

def clean_text(text):
    """
    Cleans the input text by removing non-alphanumeric characters and converting to lowercase.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = text.lower()
    return text

def build_vocab(data_file, threshold=1):
    """
    Builds vocabularies with words from the data file that appear at least `threshold` times.
    Returns two separate vocabularies, one for the original text and one for the translated text.
    """
    # Counter objects to hold the words and their frequencies for each language
    orig_word_freq = Counter()
    trans_word_freq = Counter()

    # Read the data file and update the word frequency for each language
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            original, translated = line.strip().split('####')
            original = clean_text(original)
            translated = clean_text(translated)
            orig_word_freq.update(original.split())
            trans_word_freq.update(translated.split())

    # Discard the words that appear less than the threshold number of times for each language
    orig_vocab = [word for word, freq in orig_word_freq.items() if freq >= threshold]
    trans_vocab = [word for word, freq in trans_word_freq.items() if freq >= threshold]

    # Sort the vocabularies to ensure a deterministic order
    orig_vocab.sort()
    trans_vocab.sort()

    # Add a special token for unknown words to each vocabulary
    orig_vocab.append('<unk>')
    trans_vocab.append('<unk>')

    # Add a special token for padding to each vocabulary
    orig_vocab.append('<pad>')
    trans_vocab.append('<pad>')

    # Return the vocabularies
    return orig_vocab, trans_vocab

def save_vocab(vocab, file_path):
    """
    Saves the vocabulary to a file with one word per line.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for word in vocab:
            f.write(f"{word}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build vocabulary from data file')
    parser.add_argument('data_file', type=str, help='path to data file')
    args = parser.parse_args()

    orig_vocab, trans_vocab = build_vocab(args.data_file)
    save_vocab(orig_vocab, 'orig_vocab.txt')
    save_vocab(trans_vocab, 'trans_vocab.txt')
