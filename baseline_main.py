import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# !pip install pytorch_transformers
from pytorch_transformers import AdamW  # Adam's optimization w/ fixed weight decay

from models.baseline_models import SimpleRNN, SimpleRNNWithBERTEmbeddings
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
torch.manual_seed(seed=SEED)
torch.backends.cudnn.deterministic = True

# Define hyperparameters
USE_BERT_EMBEDDING_PARAMETERS = True

PRETRAINED_MODEL_NAME = 'bert-base-cased'
NUM_EPOCHS = 50
BATCH_SIZE = 32
MAX_VOCABULARY_SIZE = 25000
MAX_TOKENIZATION_LENGTH = 512
EMBEDDING_DIM = 100
NUM_CLASSES = 2
NUM_RECURRENT_LAYERS = 1
HIDDEN_SIZE = 128
USE_BIDIRECTIONAL = True
DROPOUT_RATE = 0.20

# Initialize model
if USE_BERT_EMBEDDING_PARAMETERS:
    model = SimpleRNNWithBERTEmbeddings(pretrained_model_name_for_embeddings=PRETRAINED_MODEL_NAME,
                                        max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                        num_classes=NUM_CLASSES,
                                        num_recurrent_layers=NUM_RECURRENT_LAYERS,
                                        use_bidirectional=USE_BIDIRECTIONAL,
                                        hidden_size=HIDDEN_SIZE,
                                        dropout_rate=DROPOUT_RATE,
                                        use_gpu=True if torch.cuda.is_available() else False)

# IMPORTANT NOTE: Maximum vocabulary size should be set to be equal or larger than the maximum
# encoded (embedded) index used for any token, else the embedding matrix will not capture that token
else:
    model = SimpleRNN(pretrained_model_name_for_tokenizer=PRETRAINED_MODEL_NAME,
                      max_vocabulary_size=MAX_VOCABULARY_SIZE*4,
                      max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                      embedding_dim=EMBEDDING_DIM,
                      num_classes=NUM_CLASSES,
                      num_recurrent_layers=NUM_RECURRENT_LAYERS,
                      hidden_size=HIDDEN_SIZE,
                      use_bidirectional=USE_BIDIRECTIONAL,
                      dropout_rate=DROPOUT_RATE,
                      use_gpu=True if torch.cuda.is_available() else False)


# Acquire iterators through data loaders
train_loader = DataLoader(dataset=IMDBDataset(input_directory='aclImdb/train',
                                              tokenizer=model.get_tokenizer(),
                                              apply_cleaning=False,
                                              max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                              truncation_method='head-only',
                                              device=DEVICE),
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=IMDBDataset(input_directory='aclImdb/test',
                                             tokenizer=model.get_tokenizer(),
                                             apply_cleaning=False,
                                             max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                                             truncation_method='head-only',
                                             device=DEVICE),
                         batch_size=BATCH_SIZE,
                         shuffle=False)
# Define loss function
criterion = nn.CrossEntropyLoss()

# Define identifiers & group model parameters accordingly (check README.md for the intuition)
if USE_BERT_EMBEDDING_PARAMETERS:
    bert_learning_rate = 3e-5
    custom_learning_rate = 1e-3
    bert_identifiers = ['embeddings']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']
    grouped_model_parameters = [
            {'params': [param for name, param in model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': bert_learning_rate,
             'betas': (0.9, 0.999),
             'weight_decay': 0.01,
             'eps': 1e-8},
            {'params': [param for name, param in model.named_parameters()
                        if any(identifier in name for identifier in bert_identifiers) and
                        any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
             'lr': bert_learning_rate,
             'betas': (0.9, 0.999),
             'weight_decay': 0.0,
             'eps': 1e-8},
            {'params': [param for name, param in model.named_parameters()
                        if not any(identifier in name for identifier in bert_identifiers)],
             'lr': custom_learning_rate,
             'betas': (0.9, 0.999),
             'weight_decay': 0.0,
             'eps': 1e-8}
    ]
    # Define optimizer
    optimizer = AdamW(grouped_model_parameters)

else:
    # Define optimizer
    optimizer = optim.Adam(params=model.parameters(),
                           lr=1e-3,
                           betas=(0.9, 0.999),
                           eps=1e-8)

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
        torch.save(model.state_dict(), 'saved_models/simple-lstm-model.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_acc * 100:.2f}%')
    print(f'\tTest Loss:  {test_loss:.3f} | Test Accuracy:  {test_acc * 100:.2f}%')

