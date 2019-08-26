import logging
import random
import re
import numpy as np

from tqdm import trange

import torch
from torch.utils.data import DataLoader

from models.finetuned_models import FineTunedBert
from utils.data_utils import IMDBDataset
from utils.model_utils import get_normalized_attention
from utils.visualization_utils import visualize_attention

# Disable unwanted warning messages from pytorch_transformers
# NOTE: Run once without the line below to check if anything is wrong, here we target to eliminate
# the message "Token indices sequence length is longer than the specified maximum sequence length"
# since we already take care of it within the tokenize() function through fixing sequence length
logging.getLogger('pytorch_transformers').setLevel(logging.CRITICAL)

# Specify DEVICE
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
PRETRAINED_MODEL_NAME = 'bert-base-cased'
NUM_PRETRAINED_BERT_LAYERS = 12
NUM_ATTENTION_HEADS = 12
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

SAVED_MODEL_PATH = 'saved_models/finetuned-bert-model-12VA.pt'
APPLY_CLEANING = False
TRUNCATION_METHOD = 'head-only'
NUM_WORKERS = 0

ATTENTION_VISUALIZATION_METHOD = 'custom'  # specify which layer, head, and token yourself
LAYER_ID = 9
HEAD_ID = 6
TOKEN_ID = 0
EXCLUDE_SPECIAL_TOKENS = True  # [CLS] and [SEP]
NUM_EXAMPLES = 20

# Initialize model
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

# Load model weights & assign model to correct device
model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)

# Get tokenizer
tokenizer = model.get_tokenizer()

# Acquire test iterator through data loader
test_dataset = IMDBDataset(input_directory='data/aclImdb/test',
                           tokenizer=tokenizer,
                           apply_cleaning=APPLY_CLEANING,
                           max_tokenization_length=MAX_TOKENIZATION_LENGTH,
                           truncation_method=TRUNCATION_METHOD,
                           device=DEVICE)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=len(test_dataset),
                         shuffle=True,
                         num_workers=NUM_WORKERS)

# Get all test movie reviews from test data for attention visualizations
test_input_ids = next(iter(test_loader))[0].tolist()
print("NUMBER OF TEST EXAMPLES: %d" % len(test_input_ids))

for i in trange(NUM_EXAMPLES, desc='Attending to Test Reviews', leave=True):
    example_test_input_ids = test_input_ids[i]
    example_test_sentence = tokenizer.decode(token_ids=example_test_input_ids)

    # Extract the first component in case the tokenizer categorized the text in two >= 2 pieces
    # NOTE: This usually happens when there are multiple padding ([PAD]) tokens in the text
    if isinstance(example_test_sentence, list):
        example_test_sentence = example_test_sentence[0]

    # Remove all model-induced tags to visualize attention weights on only original tokens
    example_test_sentence = example_test_sentence.replace('[CLS]', '')
    example_test_sentence = example_test_sentence.replace('[SEP]', '')
    example_test_sentence = example_test_sentence.replace('[UNK]', '')
    example_test_sentence = example_test_sentence.replace('[PAD]', '')
    example_test_sentence = example_test_sentence.lstrip().rstrip()
    example_test_sentence = re.sub(' +', ' ', example_test_sentence)

    tokens_and_weights = get_normalized_attention(model=model,
                                                  raw_sentence=example_test_sentence,
                                                  method=ATTENTION_VISUALIZATION_METHOD,
                                                  n=LAYER_ID,
                                                  m=HEAD_ID,
                                                  k=TOKEN_ID,
                                                  exclude_special_tokens=EXCLUDE_SPECIAL_TOKENS,
                                                  normalization_method='min-max',
                                                  device=DEVICE)

    visualize_attention(window_name="Attention Visualization of " +
                                    "LAYER ID.: %d, HEAD ID.: %d, TOKEN ID.: %d on EXAMPLE ID.: %d" %
                                    (LAYER_ID, HEAD_ID, TOKEN_ID, i),
                        tokens_and_weights=tokens_and_weights)
