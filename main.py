"""
Script for training, testing, and saving finetuned, binary classification models based on pretrained
BERT parameters, for the IMDB dataset.
"""

import logging
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from models.finetuned_models import FineTunedBert
from utils.data_utils import IMDBDataset
from utils.model_utils import train, test

# Disable unwanted warning messages from pytorch_transformers
# NOTE: Run once without the line below to check if anything is wrong, here we target to eliminate
# the message "Token indices sequence length is longer than the specified maximum sequence length"
# since we already take care of it within the tokenize() function through fixing sequence length
logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("DEVICE FOUND: %s" % DEVICE)

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Define hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 32

PRETRAINED_MODEL_NAME = 'bert-base-cased'
NUM_PRETRAINED_BERT_LAYERS = 4
MAX_TOKENIZATION_LENGTH = 512
NUM_CLASSES = 2
TOP_DOWN = True
NUM_RECURRENT_LAYERS = 0
HIDDEN_SIZE = 128
REINITIALIZE_POOLER_PARAMETERS = False
USE_BIDIRECTIONAL = False
DROPOUT_RATE = 0.20
AGGREGATE_ON_CLS_TOKEN = True
CONCATENATE_HIDDEN_STATES = False

APPLY_CLEANING = False
TRUNCATION_METHOD = 'head-only'
NUM_WORKERS = 0

BERT_LEARNING_RATE = 3e-5
CUSTOM_LEARNING_RATE = 1e-3
BETAS = (0.9, 0.999)
BERT_WEIGHT_DECAY = 0.01
EPS = 1e-8

# Initialize to-be-finetuned Bert model
model = FineTunedBert(pretrained_model_name=PRETRAINED_MODEL_NAME,
                      num_pretrained_bert_layers=NUM_PRETRAINED_BERT_LAYERS,
                      max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                      num_classes=NUM_CLASSES,
                      top_down=TOP_DOWN,
                      num_recurrent_layers=NUM_RECURRENT_LAYERS,
                      use_bidirectional=USE_BIDIRECTIONAL,
                      hidden_size=HIDDEN_SIZE,
                      reinitialize_pooler_parameters=REINITIALIZE_POOLER_PARAMETERS,
                      dropout_rate=DROPOUT_RATE,
                      aggregate_on_cls_token=AGGREGATE_ON_CLS_TOKEN,
                      concatenate_hidden_states=CONCATENATE_HIDDEN_STATES,
                      use_gpu=True if torch.cuda.is_available() else False)

# Initialize train & test datasets
train_dataset = IMDBDataset(input_directory='data/aclImdb/train',
                            tokenizer=model.get_tokenizer(),
                            apply_cleaning=APPLY_CLEANING,
                            max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                            truncation_method=TRUNCATION_METHOD,
                            device=DEVICE)

test_dataset = IMDBDataset(input_directory='data/aclImdb/test',
                           tokenizer=model.get_tokenizer(),
                           apply_cleaning=APPLY_CLEANING,
                           max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                           truncation_method=TRUNCATION_METHOD,
                           device=DEVICE)

# Acquire iterators through data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define identifiers & group model parameters accordingly (check README.md for the intuition)
bert_identifiers = ['embedding', 'encoder', 'pooler']
no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': BERT_LEARNING_RATE,
         'betas': BETAS,
         'weight_decay': BERT_WEIGHT_DECAY,
         'eps': EPS},
        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': BERT_LEARNING_RATE,
         'betas': BETAS,
         'weight_decay': 0.0,
         'eps': EPS},
        {'params': [param for name, param in model.named_parameters()
                    if not any(identifier in name for identifier in bert_identifiers)],
         'lr': CUSTOM_LEARNING_RATE,
         'betas': BETAS,
         'weight_decay': 0.0,
         'eps': EPS}
]

# Define optimizer
optimizer = AdamW(grouped_model_parameters)

# Place model & loss function on GPU
model, criterion = model.to(DEVICE), criterion.to(DEVICE)

# Start actual training, check test loss after each epoch
best_test_loss = float('inf')
for epoch in range(NUM_EPOCHS):
    print("EPOCH NO: %d" % (epoch + 1))

    train_loss, train_acc = train(model=model,
                                  iterator=train_loader,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  device=DEVICE,
                                  include_bert_masks=True)

    test_loss, test_acc = test(model=model,
                               iterator=test_loader,
                               criterion=criterion,
                               device=DEVICE,
                               include_bert_masks=True)

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'saved_models/finetuned-bert-model.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')
