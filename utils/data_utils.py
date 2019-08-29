"""
Script containing various utilities related to data processing and cleaning. Includes tokenization,
text cleaning, feature extractor (token type IDs & attention masks) for BERT, and IMDBDataset.
"""

import logging
import torch
from torch.utils.data import Dataset

import os
import pickle
import re
import numpy as np
from tqdm import trange
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Setup stopwords list & word (noun, adjective, and verb) lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """Function to clean text using RegEx operations, removal of stopwords, and lemmatization."""
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(' ')]
    text = [lemmatizer.lemmatize(token, 'v') for token in text]
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    text = text.lstrip().rstrip()
    text = re.sub(' +', ' ', text)
    return text


def tokenize_and_encode(text, tokenizer, apply_cleaning=False, max_tokenization_length=512,
                        truncation_method='head-only', split_head_density=0.5):
    """
    Function to tokenize & encode a given text.

    @param (str) text: a sequence of words to be tokenized in raw string format
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (bool) apply_cleaning: whether or not to perform common cleaning operations on texts;
           note that enabling only makes sense if language of the task is English (default: False)
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (str) truncation_method: method that will be applied in case the text exceeds
           @max_tokenization_length; currently implemented methods include 'head-only', 'tail-only',
           and 'head+tail' (default: 'head-only')
    @param (float) split_head_density: weight on head when splitting between head and tail, only
           applicable if @truncation_method='head+tail' (default: 0.5)
    @return (list) input_ids: the encoded integer indexes of the given text; note that
            get_data_iterators() function converts this to a Tensor under the hood
    """
    if apply_cleaning:
        text = clean_text(text=text)

    # Tokenize and encode
    tokenized_text = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Subtract 2 ([CLS] and[SEP] tokens) to get the actual text tokenization length
    text_tokenization_length = max_tokenization_length - 2
    # Truncate sequences with the specified approach
    if len(input_ids) > text_tokenization_length:
        # i)   Head-Only Approach: Keep the first N tokens
        if truncation_method == 'head-only':
            input_ids = input_ids[:text_tokenization_length]
        # ii)  Tail-Only Approach: Keep the last N tokens
        elif truncation_method == 'tail-only':
            input_ids = input_ids[-text_tokenization_length:]
        # iii) Head+Tail Approach: Keep the first F tokens and last L tokens where F + L = N
        elif truncation_method == 'head+tail':
            head_tokenization_length = int(text_tokenization_length * split_head_density)
            tail_tokenization_length = text_tokenization_length - head_tokenization_length
            input_head_ids = input_ids[:head_tokenization_length]
            input_tail_ids = input_ids[-tail_tokenization_length:]
            input_ids = input_head_ids + input_tail_ids

    # Plug in CLS & SEP special tokens for identification of start & end points of sequences
    cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    input_ids = [cls_id] + input_ids + [sep_id]

    # Pad sequences & corresponding masks and features
    pad_id = tokenizer.convert_tokens_to_ids('[PAD]')
    if len(input_ids) < max_tokenization_length:
        padding_length = max_tokenization_length - len(input_ids)
        input_ids = input_ids + ([pad_id] * padding_length)

    # Check if input is in correct length
    # assert len(input_ids) == max_tokenization_length
    return input_ids


def get_features(input_ids, tokenizer, device):
    """
    Function to get BERT-related features, and helps to build the total input representation.

    @param (Tensor) input_ids: the encoded integer indexes of a batch, with shape: (B, P)
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (str) device: 'cpu' or 'gpu', decides where to store the outputted tensors
    @return (Tensor, Tensor) token_type_ids, attention_mask: features describe token type with
            a 0 for the first sentence and a 1 for the pair sentence; enable attention on a
            particular token with a 1 or disable it with a 0
    """
    token_type_ids, attention_mask = [], []

    # Iterate over batch
    for input_ids_example in input_ids:
        # Convert tensor to a 1D list
        input_ids_example = input_ids_example.squeeze().tolist()
        # Set example to whole input when batch size is 1
        if input_ids.shape[0] == 1:
            input_ids_example = input_ids.squeeze().tolist()
        # Get padding information
        padding_token_id = tokenizer.encode('[PAD]')[0]
        padding_length = input_ids_example.count(padding_token_id)
        text_length = len(input_ids_example) - padding_length

        # Get segment IDs -> all 0s for one sentence, which is the case for sequence classification
        token_type_ids_example = [0] * len(input_ids_example)
        # Get input mask -> 1 for real tokens, 0 for padding tokens
        attention_mask_example = ([1] * text_length) + ([0] * padding_length)

        # Check if features are in correct length
        assert len(token_type_ids_example) == len(input_ids_example)
        assert len(attention_mask_example) == len(input_ids_example)
        token_type_ids.append(token_type_ids_example)
        attention_mask.append(attention_mask_example)

    # Convert lists to tensors
    token_type_ids = torch.tensor(data=token_type_ids, device=device)
    attention_mask = torch.tensor(data=attention_mask, device=device)
    return token_type_ids, attention_mask


class IMDBDataset(Dataset):
    """
    IMDB Dataset for easily iterating over and performing common operations.

    @param (str) input_directory: path of directory where the desired data exists
    @param (pytorch_transformers.BertTokenizer) tokenizer: tokenizer with pre-figured mappings
    @param (bool) apply_cleaning: whether or not to perform common cleaning operations on texts;
           note that enabling only makes sense if language of the task is English
    @param (int) max_tokenization_length: maximum number of positional embeddings, or the sequence
           length of an example that will be fed to BERT model (default: 512)
    @param (str) truncation_method: method that will be applied in case the text exceeds
           @max_tokenization_length; currently implemented methods include 'head-only', 'tail-only',
           and 'head+tail' (default: 'head-only')
    @param (float) split_head_density: weight on head when splitting between head and tail, only
           applicable if @truncation_method='head+tail' (default: 0.5)
    @param (str) device: 'cpu' or 'gpu', decides where to store the data tensors

    """
    def __init__(self, input_directory, tokenizer, apply_cleaning, max_tokenization_length,
                 truncation_method='head-only', split_head_density=0.5, device='cpu'):
        super(IMDBDataset).__init__()
        self.positive_path = os.path.join(input_directory, 'pos')
        self.positive_files = [f for f in os.listdir(self.positive_path)
                               if os.path.isfile(os.path.join(self.positive_path, f))]
        self.num_positive_examples = len(self.positive_files)
        self.positive_label = 1
        self.negative_path = os.path.join(input_directory, 'neg')
        self.negative_files = [f for f in os.listdir(self.negative_path)
                               if os.path.isfile(os.path.join(self.negative_path, f))]
        self.num_negative_examples = len(self.negative_files)
        self.negative_label = 0

        self.tokenizer = tokenizer
        self.apply_cleaning = apply_cleaning
        self.max_tokenization_length = max_tokenization_length
        self.truncation_method = truncation_method
        self.split_head_density = split_head_density
        self.device = device

        # Pre-tokenize & encode examples
        self.pre_tokenize_and_encode_examples()

    def pre_tokenize_and_encode_examples(self):
        """
        Function to tokenize & encode examples and save the tokenized versions to a separate folder.
        This way, we won't have to perform the same tokenization and encoding ops every epoch.
        """
        if not os.path.exists(os.path.join(self.positive_path, 'tokenized_and_encoded')):
            os.mkdir(os.path.join(self.positive_path, 'tokenized_and_encoded'))

            # Clean & tokenize positive reviews
            for i in trange(len(self.positive_files), desc='Tokenizing & Encoding Positive Reviews',
                            leave=True):
                file = self.positive_files[i]
                with open(os.path.join(self.positive_path, file), mode='r', encoding='utf8') as f:
                    example = f.read()

                example = re.sub(r'<br />', '', example)
                example = example.lstrip().rstrip()
                example = re.sub(' +', ' ', example)
                example = tokenize_and_encode(text=example,
                                              tokenizer=self.tokenizer,
                                              apply_cleaning=self.apply_cleaning,
                                              max_tokenization_length=self.max_tokenization_length,
                                              truncation_method=self.truncation_method,
                                              split_head_density=self.split_head_density)

                with open(os.path.join(self.positive_path, 'tokenized_and_encoded', file), mode='wb') as f:
                    pickle.dump(obj=example, file=f)
        else:
            logging.warning('Tokenized positive reviews directory already exists!')

        if not os.path.exists(os.path.join(self.negative_path, 'tokenized_and_encoded')):
            os.mkdir(os.path.join(self.negative_path, 'tokenized_and_encoded'))

            # Clean & tokenize negative reviews
            for i in trange(len(self.negative_files), desc='Tokenizing & Encoding Negative Reviews',
                            leave=True):
                file = self.negative_files[i]
                with open(os.path.join(self.negative_path, file), mode='r', encoding='utf8') as f:
                    example = f.read()

                example = re.sub(r'<br />', '', example)
                example = example.lstrip().rstrip()
                example = re.sub(' +', ' ', example)
                example = tokenize_and_encode(text=example,
                                              tokenizer=self.tokenizer,
                                              apply_cleaning=self.apply_cleaning,
                                              max_tokenization_length=self.max_tokenization_length,
                                              truncation_method=self.truncation_method,
                                              split_head_density=self.split_head_density)

                with open(os.path.join(self.negative_path, 'tokenized_and_encoded', file), mode='wb') as f:
                    pickle.dump(obj=example, file=f)
        else:
            logging.warning('Tokenized negative reviews directory already exists!')

    def __len__(self):
        return len(self.positive_files) + len(self.negative_files)

    def __getitem__(self, index):
        if index < self.num_positive_examples:
            file = self.positive_files[index]
            label = torch.tensor(data=self.positive_label, dtype=torch.long).to(self.device)
            with open(os.path.join(self.positive_path, 'tokenized_and_encoded', file), mode='rb') as f:
                example = pickle.load(file=f)
        elif index >= self.num_positive_examples:
            file = self.negative_files[index-self.num_positive_examples]
            label = torch.tensor(data=self.negative_label, dtype=torch.long).to(self.device)
            with open(os.path.join(self.negative_path, 'tokenized_and_encoded', file), mode='rb') as f:
                example = pickle.load(file=f)
        else:
            raise ValueError('Out of range index while accessing dataset')

        return torch.from_numpy(np.array(example)).long().to(self.device), label
