import json
import numpy as np

class RnaTokenizer:
    def __init__(self):
        # Define the fixed RNA vocabulary and special tokens.
        self.special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.base_vocab = ['A', 'U', 'C', 'G', 'N','T','R','Y','S','W','K','M','B','D','H','V']
        self.ex_vocab=['<mask>']
        # 2. Build the complete vocabulary.
        self.vocab = {}
        # First, add the special tokens.
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        # Then, add the RNA bases.
        for base in self.base_vocab:
            self.vocab[base] = len(self.vocab)
        for ex in self.ex_vocab:
            self.vocab[ex] = len(self.vocab)

        # 3. Create the inverse vocabulary for decoding.
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 4. Define special token IDs for convenient access.
        self.unk_idx = self.vocab["<unk>"]
        self.pad_idx = self.vocab["<pad>"]
        self.bos_idx = self.vocab["<bos>"]
        self.eos_idx = self.vocab["<eos>"]
        self.mask_idx = self.vocab.get("<mask>", None)
        
    @property
    def vocab_size(self) -> int:
        """Returns the total vocabulary size."""
        return len(self.vocab)
        
    def encode(self, seqs: list[str], add_special_tokens: bool = True) -> list[dict]:
        """
        Encodes a batch of RNA strings into lists of token IDs.
        Returns a list of dictionaries to maintain compatibility with the original code format.
        """
        encoded_seqs = []
        for s in seqs:
            # 1. Convert each character in the string to its ID.
            ids = [self.vocab.get(char.upper(), self.unk_idx) for char in s]
            
            if add_special_tokens:
                ids = [self.bos_idx] + ids + [self.eos_idx]

            encoded_seqs.append({
                'input_ids': np.array(ids, dtype=np.int32)
            })
            
        return encoded_seqs

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decodes a single list of token IDs back into an RNA string."""
        chars = []
        for token_id in token_ids:
            char = self.inverse_vocab.get(token_id, "<unk>")
   
            # if skip_special_tokens and char in self.special_tokens:
            #     continue
            chars.append(char)
            
        return "".join(chars)

    def save(self, file_path: str):
        """Saves the tokenizer configuration (vocabulary) to a JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"RNA Tokenizer config saved to {file_path}")

    @classmethod
    def load(cls, file_path: str):
        """Loads the tokenizer configuration from a JSON file."""
        tokenizer = cls()
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_vocab = json.load(f)
        
        # Validate and reconstruct the instance.
        # Ensure the loaded vocabulary is consistent with the class structure.
        tokenizer.vocab = loaded_vocab
        tokenizer.inverse_vocab = {v: k for k, v in loaded_vocab.items()}
        tokenizer.special_tokens = []
        tokenizer.base_vocab = []
        tokenizer.ex_vocab = []

        tokenizer.unk_idx = tokenizer.vocab.get("<unk>", 0)
        tokenizer.pad_idx = tokenizer.vocab.get("<pad>", 1)
        tokenizer.bos_idx = tokenizer.vocab.get("<bos>", 2)
        tokenizer.eos_idx = tokenizer.vocab.get("<eos>", 3)
        return tokenizer