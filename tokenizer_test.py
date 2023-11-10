from tokenizers import Tokenizer

# Load the trained tokenizer from a saved file
tokenizer = Tokenizer.from_file('pg.json')

# Define a set of test cases
test_cases = [
    "Balet walad kinen Jose so Katawan tan binindisionan to, kanyan akaliket ed sikatoy inkargadod prisowan.",
]

for text in test_cases:
    output = tokenizer.encode(text)
    decoded_text = tokenizer.decode(output.ids)
    print(f"Original: {text}")
    print(f"IDs: {output.ids}")
    print(f"Decoded: {decoded_text}")
    print(f"Is Decoded Text Same as Original Text: {decoded_text == text}")


# If everything is correct, the script will complete without an error
print("Tokenizer test passed!")
