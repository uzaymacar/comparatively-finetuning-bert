"""
Class file for baseline models, conventional many-to-one architectures, for comparison. For ease of
notation, the following abbreviations are used in comments next to some tensor operations:
i)    B  = batch size,
ii)   P  = maximum number of positional embeddings from BERT tokenizer (default: 512),
iii)  H  = hidden size dimension in pretrained BERT layers (default: 768),
iv)   H* = hidden size dimension for the additional recurrent (LSTM) layer,
v)    L  = number of recurrent layers
"""

import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_transformers import BertConfig, BertTokenizer, BertModel


class SimpleRNN(nn.Module):
    """
    Simple model that utilizes BERT tokenizer, custom embedding, a recurrent neural network choice
    of LSTM, dropout, and finally a dense layer for classification.

    @param (str) pretrained_model_name_for_tokenizer: name of the pretrained BERT model for
           tokenizing input sequences
    @param (int) max_vocabulary_size: upper limit for number of tokens in the embedding layer
    @param (int) max_tokenization_length: number of tokens to pad / truncate input sequences to
    @param (int) embedding_dim: dimension size of each token representation for the embedding layer
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """
    def __init__(self, pretrained_model_name_for_tokenizer, max_vocabulary_size,
                 max_tokenization_length, embedding_dim, num_classes=1, num_recurrent_layers=1,
                 use_bidirectional=False, hidden_size=128, dropout_rate=0.10, use_gpu=False):
        super(SimpleRNN, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu

        # Configure tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_for_tokenizer)
        self.tokenizer.max_len = max_tokenization_length

        # Define additional layers & utilities specific to the finetuned task
        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=max_vocabulary_size,
                                      embedding_dim=embedding_dim)

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # Recurrent Layer
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=num_recurrent_layers,
                            bidirectional=use_bidirectional,
                            batch_first=True)

        # Dense Layer for Classification
        self.clf = nn.Linear(in_features=hidden_size*2 if use_bidirectional else hidden_size,
                             out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def forward(self, input_ids, token_type_ids=None,                          # input_ids: (B, P)
                attention_mask=None, position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        embedded_output = self.embedding(input_ids)

        # Set initial states
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_recurrent_layers*2              # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_recurrent_layers*2              # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size))

        # Apply recurrent layer
        lstm_output = self.lstm(embedded_output, (h0, c0))                     # (B, P, H*), (2 x (B, B, H*))
        sequence_output, _ = lstm_output

        # Get last timesteps for each example in the batch; we do this to counteract padding
        last_timesteps = []
        for i in range(len(attention_mask)):
            last_timesteps.append(
                attention_mask[i].tolist().index(0)
                if 0 in attention_mask[i].tolist() else self.tokenizer.max_len-1
            )

        if self.use_gpu:
            last_timesteps = torch.tensor(data=last_timesteps).cuda()         # (B)
        else:
            last_timesteps = torch.tensor(data=last_timesteps)                # (B)
        relative_hidden_size = self.hidden_size*2 if self.use_bidirectional else self.hidden_size
        last_timesteps = last_timesteps.repeat(1, relative_hidden_size)       # (1, B x H*)
        last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)     # (B, 1, H*)
        pooled_sequence_output = sequence_output.gather(                      # (B, H*)
            dim=1,
            index=last_timesteps
        ).squeeze()

        pooled_sequence_output = self.dropout(pooled_sequence_output)          # (B, H*)
        logits = self.clf(pooled_sequence_output)                              # (B, num_classes)
        return logits


class SimpleRNNWithBERTEmbeddings(nn.Module):
    """
    Simple model that utilizes BERT tokenizer, pretrained BERT embedding, a recurrent neural network
    choice of LSTM, dropout, and finally a dense layer for classification.

    @param (str) pretrained_model_name_for_embeddings: name of the pretrained BERT model for
           both tokenizing input sequences and extracting vector representations for each token
    @param (int) max_tokenization_length: number of tokens to pad / truncate input sequences to
    @param (int) num_classes: number of classes to distinct between for classification; specify
           2 for binary classification (default: 1)
    @param (int) num_recurrent_layers: number of LSTM layers to utilize (default: 1)
    @param (bool) use_bidirectional: whether to use a bidirectional LSTM or not (default: False)
    @param (int) hidden_size: number of recurrent units in each LSTM cell (default: 128)
    @param (float) dropout_rate: possibility of each neuron to be discarded (default: 0.10)
    @param (bool) use_gpu: whether to utilize GPU (CUDA) or not (default: False)
    """
    def __init__(self, pretrained_model_name_for_embeddings, max_tokenization_length,
                 num_classes=1, num_recurrent_layers=1, use_bidirectional=False,
                 hidden_size=128, dropout_rate=0.10, use_gpu=False):
        super(SimpleRNNWithBERTEmbeddings, self).__init__()
        self.num_recurrent_layers = num_recurrent_layers
        self.use_bidirectional = use_bidirectional
        self.hidden_size = hidden_size
        self.use_gpu = use_gpu

        # Configure tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_for_embeddings)
        self.tokenizer.max_len = max_tokenization_length

        # Define additional layers & utilities specific to the finetuned task
        # Embedding Layer
        # Get global BERT config
        self.config = BertConfig.from_pretrained(pretrained_model_name_for_embeddings)
        # Extract all parameters (weights and bias matrices) for the 12 layers
        all_states_dict = BertModel.from_pretrained(pretrained_model_name_for_embeddings,
                                                    config=self.config).state_dict()

        # Get customized BERT config
        self.config.max_position_embeddings = max_tokenization_length
        self.config.num_hidden_layers = 0
        self.config.output_hidden_states = True

        # Get pretrained BERT model & all its learnable parameters
        self.bert = BertModel.from_pretrained(pretrained_model_name_for_embeddings,
                                              config=self.config)
        current_states_dict = self.bert.state_dict()

        # Assign matching parameters (weights and biases of ONLY embeddings)
        for param in current_states_dict.keys():
            if 'embedding' in param:
                current_states_dict[param] = all_states_dict[param]

        # Update parameters in extracted BERT model
        self.bert.load_state_dict(current_states_dict)

        logging.info('Loaded %d learnable parameters from pretrained BERT model with %d layer(s)' %
                     (len(list(self.bert.parameters())), 0))

        # Dropout to prevent overfitting
        self.dropout = nn.Dropout(p=dropout_rate)

        # Recurrent Layer
        self.lstm = nn.LSTM(input_size=self.config.hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_recurrent_layers,
                            bidirectional=use_bidirectional,
                            batch_first=True)

        # Dense Layer for Classification
        self.clf = nn.Linear(in_features=hidden_size * 2 if self.use_bidirectional else hidden_size,
                             out_features=num_classes)

    def get_tokenizer(self):
        """Function to easily access the BERT tokenizer"""
        return self.tokenizer

    def forward(self, input_ids, token_type_ids=None,                          # input_ids: (B, P)
                attention_mask=None, position_ids=None, head_mask=None):
        """Function implementing a forward pass of the model"""
        # Pass tokenized sequence through pretrained BERT model & extract ONLY embeddings
        embedded_output = self.bert.embeddings(input_ids=input_ids,            # (B, P, H)
                                               position_ids=position_ids,
                                               token_type_ids=token_type_ids)

        # Set initial states
        if self.use_gpu:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size)).cuda()
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size))
            c0 = Variable(torch.zeros(self.num_recurrent_layers * 2            # (L * 2 OR L, B, H)
                                      if self.use_bidirectional else self.num_recurrent_layers,
                                      input_ids.shape[0],
                                      self.hidden_size))

        lstm_output = self.lstm(embedded_output, (h0, c0))                     # (B, P, H*), (2 x (B, B, H*))
        sequence_output, _ = lstm_output

        # Get last timesteps for each example in the batch; we do this to counteract padding
        last_timesteps = []
        for i in range(len(attention_mask)):
            last_timesteps.append(
                attention_mask[i].tolist().index(0)
                if 0 in attention_mask[i].tolist() else self.tokenizer.max_len - 1
            )

        if self.use_gpu:
            last_timesteps = torch.tensor(data=last_timesteps).cuda()          # (B)
        else:
            last_timesteps = torch.tensor(data=last_timesteps)                 # (B)
        relative_hidden_size = self.hidden_size*2 if self.use_bidirectional else self.hidden_size
        last_timesteps = last_timesteps.repeat(1, relative_hidden_size)        # (1, B x H*)
        last_timesteps = last_timesteps.view(-1, 1, relative_hidden_size)      # (B, 1, H*)
        pooled_sequence_output = sequence_output.gather(                       # (B, H*)
            dim=1,
            index=last_timesteps
        ).squeeze()

        pooled_sequence_output = self.dropout(pooled_sequence_output)          # (B, H*)
        logits = self.clf(pooled_sequence_output)                              # (B, num_classes)
        return logits
