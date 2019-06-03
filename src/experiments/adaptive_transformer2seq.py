import os.path
import datetime
import torch
import torch.nn
import tqdm

import allennlp.data
from allennlp.data.iterators import BucketIterator

import allennlp.training

import allennlp
import allennlp.common
import allennlp.models
import allennlp.modules
import allennlp.predictors
from allennlp.modules.attention import BilinearAttention, DotProductAttention

import config

import data_adapter
import utils.opt_parser
from models.adaptive_seq2seq import AdaptiveSeq2Seq
from models.transformer.multi_head_attention import SingleTokenMHAttentionWrapper, GeneralMultiHeadAttention
from utils.nn import AllenNLPAttentionWrapper
from models.transformer.encoder import TransformerEncoder
from models.adaptive_rnn_cell import ACTRNNCell
from models.universal_hidden_state_wrapper import UniversalHiddenStateWrapper, RNNType
from models.stacked_rnn_cell import StackedLSTMCell, StackedGRUCell
from allennlp.common.util import START_SYMBOL, END_SYMBOL

def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--emb-dim', type=int, help='basic embedding dimension')
    parser.add_argument('--act-max-layer', type=int, help='maximum number of stacked layers')
    parser.add_argument('--use-act', action="store_true", help='Use adaptive computation time for decoder')
    parser.add_argument('--act-loss-weight', type=float, help="the loss of the act weights")

    parser.add_argument('--enc-layers', type=int, help="layers in encoder")
    parser.add_argument('--act-mode', choices=['basic', 'random', 'mean_field'])
    parser.add_argument('--encoder', choices=['transformer', 'lstm', 'bilstm'])
    parser.add_argument('--decoder', choices=['lstm', 'rnn', 'gru', 'ind_rnn', 'n_lstm', 'n_gru'], )
    parser.add_argument('--dec-cell-height', type=int, help="the height for n_layer lstm/gru")

    args = parser.parse_args()

    reader = data_adapter.GeoQueryDatasetReader()
    training_set = reader.read(config.DATASETS[args.dataset].train_path)
    try:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    if args.epoch:
        config.TRAINING_LIMIT = args.epoch
    if args.device:
        config.DEVICE = args.device
    st_ds_conf = get_updated_settings(args)

    model = get_model(vocab, st_ds_conf)

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=st_ds_conf['batch_sz'])
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters(), lr=config.ADAM_LR, betas=config.ADAM_BETAS, eps=config.ADAM_EPS)

        savepath = os.path.join(config.SNAPSHOT_PATH, args.dataset, 'ada_trans2seq',
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + "--" + args.memo)
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)

        trainer = allennlp.training.Trainer(
            model=model,
            optimizer=optim,
            iterator=iterator,
            train_dataset=training_set,
            validation_dataset=validation_set,
            serialization_dir=savepath,
            cuda_device=config.DEVICE,
            num_epochs=config.TRAINING_LIMIT,
            grad_clipping=config.GRAD_CLIPPING,
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        if config.DEVICE > -1:
            model = model.cuda(config.DEVICE)

        predictor = allennlp.predictors.SimpleSeq2SeqPredictor(model, reader)

        for instance in tqdm.tqdm(testing_set, total=len(testing_set)):
            print('SRC: ', instance.fields['source_tokens'].tokens)
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            print('PRED:', ' '.join(output['predicted_tokens']))

def get_model(vocab, st_ds_conf):
    emb_sz = st_ds_conf['emb_sz']

    source_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('nltokens'),
                                                  embedding_dim=emb_sz)
    target_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('lftokens'),
                                                  embedding_dim=emb_sz)

    if st_ds_conf['encoder'] == 'lstm':
        encoder = allennlp.modules.seq2seq_encoders.PytorchSeq2SeqWrapper(
            torch.nn.LSTM(emb_sz, emb_sz, st_ds_conf['num_enc_layers'], batch_first=True)
        )
    elif st_ds_conf['encoder'] == 'bilstm':
        encoder = allennlp.modules.seq2seq_encoders.PytorchSeq2SeqWrapper(
            torch.nn.LSTM(emb_sz, emb_sz, st_ds_conf['num_enc_layers'], batch_first=True, bidirectional=True)
        )
    elif st_ds_conf['encoder'] == 'transformer':
        encoder = TransformerEncoder(input_dim=emb_sz,
                                     num_layers=st_ds_conf['num_enc_layers'],
                                     num_heads=st_ds_conf['num_heads'],
                                     feedforward_hidden_dim=emb_sz,
                                     feedforward_dropout=st_ds_conf['feedforward_dropout'],
                                     residual_dropout=st_ds_conf['residual_dropout'],
                                     attention_dropout=st_ds_conf['attention_dropout'],
                                     )
    else:
        assert False

    enc_out_dim = encoder.get_output_dim()
    dec_out_dim = emb_sz

    dwa = get_attention(st_ds_conf, st_ds_conf['dwa'])
    dec_hist_attn = get_attention(st_ds_conf, st_ds_conf['dec_hist_attn'])
    enc_attn = get_attention(st_ds_conf, st_ds_conf['enc_attn'])
    if st_ds_conf['enc_attn'] == 'dot_product':
        assert enc_out_dim == dec_out_dim, "encoder hidden states must be able to multiply with decoder output"

    def sum_attn_dims(attns, dims):
        return sum(dim for attn, dim in zip(attns, dims) if attn is not None)

    dec_in_dim = dec_out_dim + 1 + sum_attn_dims([enc_attn, dec_hist_attn],
                                                 [enc_out_dim, dec_out_dim])
    rnn_cell = get_rnn_cell(st_ds_conf, dec_in_dim, dec_out_dim)

    halting_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn, dwa],
                                                 [enc_out_dim, dec_out_dim, dec_out_dim])
    halting_fn = torch.nn.Sequential(
        # torch.nn.Dropout(st_ds_conf['act_dropout']),
        torch.nn.Linear(halting_in_dim, halting_in_dim),
        torch.nn.ReLU(),
        torch.nn.Dropout(st_ds_conf['act_dropout']),
        torch.nn.Linear(halting_in_dim, 1),
        torch.nn.Sigmoid(),
    )

    decoder = ACTRNNCell(rnn_cell=rnn_cell,
                         halting_fn=halting_fn,
                         use_act=st_ds_conf['act'],
                         act_max_layer=st_ds_conf['act_max_layer'],
                         act_epsilon=st_ds_conf['act_epsilon'],
                         rnn_input_dropout=st_ds_conf['decoder_dropout'],
                         depth_wise_attention=dwa,
                         state_mode=st_ds_conf['act_mode'],
                         )
    proj_in_dim = dec_out_dim + sum_attn_dims([enc_attn, dec_hist_attn], [enc_out_dim, dec_out_dim])
    word_proj = torch.nn.Linear(proj_in_dim, vocab.get_vocab_size('lftokens'))
    model = AdaptiveSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            word_projection=word_proj,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            target_namespace='lftokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=st_ds_conf['max_decoding_len'],
                            enc_attention=enc_attn,
                            dec_hist_attn=dec_hist_attn,
                            act_loss_weight=st_ds_conf['act_loss_weight'],
                            prediction_dropout=st_ds_conf['prediction_dropout'],
                            embedding_dropout=st_ds_conf['embedding_dropout'],
                            )
    return model

def get_updated_settings(args):
    st_ds_conf = config.ADA_TRANS2SEQ_CONF[args.dataset]
    if args.act_max_layer:
        st_ds_conf['act_max_layer'] = args.act_max_layer
    if args.batch:
        st_ds_conf['batch_sz'] = args.batch
    if args.use_act:
        st_ds_conf['act'] = True
    if args.emb_dim:
        st_ds_conf['emb_sz'] = args.emb_dim
    if args.act_loss_weight:
        st_ds_conf['act_loss_weight'] = args.act_loss_weight
    if args.act_mode:
        st_ds_conf['act_mode'] = args.act_mode
    if args.enc_layers:
        st_ds_conf['num_enc_layers'] = args.enc_layers
    if args.encoder:
        st_ds_conf['encoder'] = args.encoder
    if args.decoder:
        st_ds_conf['decoder'] = args.decoder
    if args.dec_cell_height:
        st_ds_conf['dec_cell_height'] = args.dec_cell_height
    return st_ds_conf

def get_rnn_cell(st_ds_conf: dict, input_dim: int, hidden_dim: int):
    cell_type = st_ds_conf['decoder']
    if cell_type == "lstm":
        return UniversalHiddenStateWrapper(RNNType.LSTM(input_dim, hidden_dim))
    elif cell_type == "gru":
        return UniversalHiddenStateWrapper(RNNType.GRU(input_dim, hidden_dim))
    elif cell_type == "ind_rnn":
        return UniversalHiddenStateWrapper(RNNType.IndRNN(input_dim, hidden_dim))
    elif cell_type == "rnn":
        return UniversalHiddenStateWrapper(RNNType.VanillaRNN(input_dim, hidden_dim))
    elif cell_type == 'n_lstm':
        n_layer = st_ds_conf['dec_cell_height']
        return StackedLSTMCell(input_dim, hidden_dim, n_layer, st_ds_conf['decoder_dropout'])
    elif cell_type == 'n_gru':
        n_layer = st_ds_conf['dec_cell_height']
        return StackedGRUCell(input_dim, hidden_dim, n_layer, st_ds_conf['decoder_dropout'])
    else:
        raise ValueError(f"RNN type of {cell_type} not found.")

def get_attention(st_ds_conf, attn_type):
    emb_sz = st_ds_conf['emb_sz']   # dim for both the decoder output and the encoder output
    attn_type = attn_type.lower()
    if attn_type == "bilinear":
        attn = BilinearAttention(vector_dim=emb_sz, matrix_dim=emb_sz)
        attn = AllenNLPAttentionWrapper(attn, st_ds_conf['attention_dropout'])
    elif attn_type == "dot_product":
        attn = DotProductAttention()
        attn = AllenNLPAttentionWrapper(attn, st_ds_conf['attention_dropout'])
    elif attn_type == "multihead":
        attn = GeneralMultiHeadAttention(num_heads=st_ds_conf['num_heads'],
                                         input_dim=emb_sz,
                                         total_attention_dim=emb_sz,
                                         total_value_dim=emb_sz,
                                         attend_to_dim=emb_sz,
                                         output_dim=emb_sz,
                                         attention_dropout=st_ds_conf['attention_dropout'],
                                         use_future_blinding=False,
                                         )
        attn = SingleTokenMHAttentionWrapper(attn)
    elif attn_type == "none":
        attn = None
    else:
        assert False

    return attn

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

