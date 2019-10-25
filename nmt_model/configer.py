# Configuration file making using Sacred TODO

import numpy as np 
from sacred import Experiment
ex = Experiment()

@ex.config
def cfg():
    nr_layers = 10
    alpha = -1.0/ nr_layers**(1/2)
    opt_type = 'adam'
    lr = 0.1 if opt_type == 'sgd' else 0.001
    seed=2019

@ex.capture
def create_optimizer(opt_type, learning_rate):
    # Use this config parameters
    return 0

@ex.automain
def plan_R():
    return np.random.rand()


"""
Experiment for OpenNMT based sequence transduction
"""

from sacred import Experiment
from sacred.observers import MongoObserver
import onmt
import torch

# Experiment Metadata and Configuration

ex_name='gp-nlp'
ex = Experiment(ex_name)
ex.observers.append(MongoObserver(db_name=ex_name))


# Global enums
coder_types = ["brnn", "transformer", "cnn"]
rnn_types  = ["LSTM", "GRU", "SRU"]
attn_types = ["general", "dot", "mlp", "none"]
optim_types = ["sgd","adam"]

@ex.config
def cfg():

    # setup
    seed = 2019 # Random seed 
    data = TODO # Path prefix to the “.train.pt” and “.valid.pt” file path from preprocess.sh
    save_moodel = TODO # Model filename (the model will be saved as <save_model>_N.pt where N is the number of steps
    model_type = "text" # Text-based seq2seq
    model_dtype = "fp32" # Precision datatype [fp32|fp16]

    # emb
    emb_size = TODO # Source and Target emb size
    coder_type = 'rnn'
    vocab_size = TODO # Size fof the vocabluary

    # encoder
    encoder_type = "rnn" # [rnn|brnn|transformer|cnn]
    enc_layers = "2" # number of encoder layers
    enc_size = 500 # Hidden layer size for encoder
    cnn_kernel_width = 3 # CNN kernel size 
    rnn_type = "LSTM" # [LSTM|GRU|SRU]
    self_attn_type = 'scaled-dot' # [scaled-dot|average]
    max_relative_positions = 0 # Read https://arxiv.org/pdf/1803.02155.pdf
    heads = 8 # Number of Attention Heads
    transformer_ff = 2048 # Feed-forward transformer
    copy_attn = True  # Train copy attention
    copy_attn_type = 'general' # [dot|general|mlp|none]
    
    # decoder
    decoder_type = "rnn" # [rnn|brnn|transformer|cnn]
    dec_layers = "2" # number of decoder layers
    dec_size = 500 # Hidden layer size for decoder
    input_feed = 1 # Perform input feeding

    # attention
    global_attention = "general"
    # init
    param_init_glorot = False #Init parameters with xavier_uniform. Required for transformer.
    train_from = "" # If training from a checkpoint then this is the path to the pretrained model’s state_dict.
    reset_optim = "none" # [none|all|states|keep_states] ; Optimization resetter when train_from
    fix_word_vecs_enc = False # Fix word embding on encoder
    fix_word_vecs_dec = False # Fix word embding on decoder
    # loggging
    save_checkpoint_steps = 5000 # Save a checkpoint every X steps
    log_freq = 50 # Print stats at this interval.
    log_file = '' # Output logs to a file under this path.
    # optim
    optim = 'sgd' # [sgd,adam]
    learning_rate = 1 # 
    momentum = 0
    learning_rate_decay = 0.5 # update_learning_rate, decay learning rate by this much if steps have gone past
    beta1 = 0.9
    beta2 = 0.999

    # trainer
    batch_size = 64 # Batch size
    batch_type = 'sents' # [sents|tokens] ; Token gived dynamic batching
    norm = 'sents' # [sents|tokens] ; Normalization-level
    valid_steps = 10000 # Validation steps
    valid_batch_size = batch_size
    train_steps = 10000 # Training Steps
    epochs = 1 # epoch count
    early_stopping = 5 # Early stopping number
    dropout = 0.3 # LSTM dropouts
    attention_dropout = 0.1 # Attention dropout
    start_decay_steps = 50000 # Start decaying every decay_steps after start_decay_steps
    decay_steps = 10000 # Decay every decay_steps
    decay_method = 'none' # [noam|noamwd|rsqrt|none]
    warmup_steps = 4000 # Number of warmup steps for custom decay.


@ex.capture
def emb(
    coder_type
    emb_size,
    vocab_size):

    if coder_type != 'brnn':
        
        return onmt.modules.Embeddings(
            word_vec_size = emb_size,
            word_vocab_size = vocab_size,
            word_padding_idx = 0,
            position_encoding=True)
    else:
        return onmt.modules.Embeddings(
            word_vec_size = emb_size,
            word_vocab_size = vocab_size,
            word_padding_idx = 0)

@ex.capture
def optim(
    model, 
    beta1,
    beta2,
    batch_size,
    batch_type,
    norm,
    valid_steps,
    valid_batch_size,
    train_steps,
    epochs,
    early_stopping,
    optim,
    dropout,
    attention_dropout,
    learning_rate,
    learning_rate_decay,
    start_decay_steps,
    decay_steps,
    decay_method,
    warmup_steps):

    assert optim in optim_types

    if optim == 'sgd':

        return torch.optim.SGD(
            params = model.params(),
            lr = learning_rate,
            momentum=momentum)

    elif optim == 'adam':

        return torch.optim.Adam(
            params = model.params(),
            lr = learning_rate,
            betas=(beta1,beta2),
            weight_decay=learning_rate_decay,
        )

@ex.capture
def encoder(
    emb,
    encoder_type,
    rnn_type,
    enc_layers,
    enc_size,
    dropout,
    cnn_kernel_width,
    heads,
    feed_forward_size
    ):

    assert encoder_type in coder_types
    assert if encoder_type == 'rnn': rnn_type in rnn_types else: True

    if  encoder_type == 'brnn':

        return onmt.encoders.RNNEncoder(
            rnn_type = rnn_type,
            bidirectional = encoder_type == 'brnn',
            num_layers = enc_layers,
            hidden_size = enc_size,
            dropout=dropout,
            embeddings=emb)

    elif encoder_type == "cnn":

        return onmt.encoders.CNNEncoder(
            num_layers = enc_layers,
            hidden_size = enc_size,
            cnn_kernel_width = cnn_kernel_width,
            dropout = dropout,
            embeddings = emb)

    elif encoder_type == "transformer":

        return onmt.encoders.TransformerEncoder(
            num_layers = enc_layers,
            d_model = enc_layers,
            heads = heads,
            d_ff = feed_forward_size,
            embeddings = emb)

@ex.capture
def decoder(
    emb,
    encoder
    decoder_type,
    rnn_type,
    dec_layers,
    dec_size,
    attn_type,
    dropout,
    input_feed,
    cnn_kernel_width,
    heads,
    feed_forward_size
    self_attn_type,
    attention_dropout,
    copy_attn,
    copy_attn_type
    ):

    assert decoder_type in coder_types
    assert if decoder_type == 'rnn': rnn_type in rnn_types else: True

    if decoder_type == 'brnn':

        if input_feed == True:

            return onmt.decoders.InputFeedRNNDecoder(
                rnn_type = rnn_type,
                bidirectional_encoder = encoder,
                num_layers = dec_layers,
                hidden_size = dec_size,
                attn_type = attn_type,
                copy_attn=copy_attn,
                copy_attn_type=copy_attn_type,
                dropout = dropout)

        else:

            return onmt.decoders.StdRNNDecoder(
                rnn_type = rnn_type,
                bidirectional_encoder = encoder,
                num_layers = dec_layers,
                hidden_size = dec_size,
                attn_type = attn_type,
                copy_attn=copy_attn,
                copy_attn_type=copy_attn_type,
                dropout = dropout)

    elif decoder_type == "cnn":

        return onmt.decoders.CNNDecoder(
            num_layers = dec_layers,
            hidden_size = dec_size,
            attn_type = attn_type,
            cnn_kernel_width = cnn_kernel_width,
            dropout = dropout,
            copy_attn=copy_attn,
            copy_attn_type=copy_attn_type,
            embeddings = emb)
            
    elif decoder_type == "transformer":

        return onmt.decoders.TransformerDecoder(
            num_layers = dec_layers,
            d_model = dec_layers,
            heads = heads,
            d_ff = feed_forward_size,
            self_attn_type = self_attn_type
            dropout = dropout,
            copy_attn=copy_attn,
            attention_dropout = attention_dropout,
            copy_attn_type=copy_attn_type,
            embeddings = emb)

def init():


@ex.capture
def model():

    return onmt.models.NMTModel(encoder(emb()), decoder(emb()))

@ex.capture
def trainer(
    data
    save_moodel
    model_dtype 
    emb_size
    share_embeddings
    position_encoding
    encoder_type
    enc_layers
    enc_rnn_size
    cnn_kernel_width
    rnn_type
    self_attn_type
    
    heads
    transformer_ff
    aan_useffn
    copy_attn
    copy_attn_type
    generator_function
    copy_attn_force
    reuse_copy_attn
    decoder_type
    dec_layers
    dec_rnn_size
    input_feed
    global_attention,
    global_attention_fn,
    param_init_glorot,
    train_from,
    reset_optim,
    fix_word_vecs_enc,
    fix_word_vecs_dec,
    save_checkpoint_steps
    log_freq):

    model = model()

    train_loss = onmt.utils.loss.NMTLossCompute()

@ex.automain
def run():
    pass








    



