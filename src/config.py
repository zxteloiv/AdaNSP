# config

import os.path


# ======================
# general config

DEVICE = 0
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SNAPSHOT_PATH = os.path.join(ROOT, 'snapshots')

LOG_REPORT_INTERVAL = (1, 'iteration')
TRAINING_LIMIT = 500  # in num of epochs
SAVE_INTERVAL = (100, 'iteration')

ADAM_LR = 1e-3
ADAM_BETAS = (.9, .98)
ADAM_EPS = 1e-9

GRAD_CLIPPING = 5

SGD_LR = 1e-2

# ======================
# dataset config

DATA_PATH = os.path.join(ROOT, 'data')

from utils.dataset_path import DatasetPath

DS_ATIS = 'atis'
DS_SQA = 'sqa'
DS_GEOQUERY = 'geoqueries'
DS_GEOQUERY_SP = 'geoqueries_sp'
DS_WIKISQL = 'wikisql'
DS_DJANGO = 'django'

DATASETS = dict([
    (DS_ATIS, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_ATIS, 'train.json'),
        dev_path=os.path.join(DATA_PATH, DS_ATIS, 'dev.json'),
        test_path=os.path.join(DATA_PATH, DS_ATIS, 'test.json'),
    )),

    (DS_SQA, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_SQA, 'train.tsv'),
        dev_path=os.path.join(DATA_PATH, DS_SQA, 'dev.tsv'),
        test_path=os.path.join(DATA_PATH, DS_SQA, 'test.tsv'),
    )),

    (DS_GEOQUERY_SP, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_GEOQUERY, 'train.json'),
        dev_path=os.path.join(DATA_PATH, DS_GEOQUERY, 'dev.json'),
        test_path=os.path.join(DATA_PATH, DS_GEOQUERY, 'test.json'),
    )),

    (DS_GEOQUERY, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_GEOQUERY, 'orig.train.json'),
        dev_path="",
        test_path=os.path.join(DATA_PATH, DS_GEOQUERY, 'test.json'),
    )),

    (DS_WIKISQL, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_WIKISQL, 'train.json'),
        dev_path="",
        test_path=os.path.join(DATA_PATH, DS_WIKISQL, 'test.json'),
    )),

    (DS_DJANGO, DatasetPath(
        train_path=os.path.join(DATA_PATH, DS_DJANGO, 'train.json'),
        dev_path="",
        test_path=os.path.join(DATA_PATH, DS_DJANGO, 'test.json'),
    )),

])

# ======================
# setting config

ST_SEQ2SEQ = 'seq2seq'
SEQ2SEQ_CONF = dict()
SEQ2SEQ_CONF[DS_GEOQUERY] = dict(
    emb_sz=50,
    batch_sz=32,
    max_decoding_len=50,
)
SEQ2SEQ_CONF[DS_ATIS] = dict(
    emb_sz=200,
    batch_sz=32,
    max_decoding_len=60,

)

ST_TRANS2SEQ = 'transformer2seq'
TRANS2SEQ_CONF = dict()
TRANS2SEQ_CONF[DS_GEOQUERY] = dict(
    emb_sz=256,
    batch_sz=32,
    max_decoding_len=50,
)
TRANS2SEQ_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=32,
    max_decoding_len=50,
    num_heads=8,
    max_num_layers=1,
    act=False,
    residual_dropout=.1,
    attention_dropout=.1,
    feedforward_dropout=.1,
    vanilla_wiring=False,
)

ST_BASE_S2S = 'base_s2s'
BASE_S2S_CONF = dict()
BASE_S2S_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=20,
    max_decoding_len=60,
    num_heads=8,
    num_enc_layers=2,
    encoder='lstm',
    decoder='lstm',
    residual_dropout=.1,
    attention_dropout=.1,
    feedforward_dropout=.1,
    intermediate_dropout=.5,
    vanilla_wiring=False,
    enc_attn="dot_product",
    dec_hist_attn="dot_product",
    dec_cell_height=2,
    concat_attn_to_dec_input=True,
)

ST_UNC_S2S = 'unc_s2s'
UNC_S2S_CONF = dict()
UNC_S2S_CONF[DS_GEOQUERY] = dict(
    emb_sz=256,
    batch_sz=20,
    max_decoding_len=60,
    num_heads=2,
    num_enc_layers=2,
    encoder='bilstm',
    residual_dropout=.1,
    attention_dropout=.1,
    feedforward_dropout=.1,
    intermediate_dropout=.5,
    vanilla_wiring=True,
    decoder='n_lstm',
    enc_attn="bilinear",
    dec_hist_attn="dot_product",
    dec_cell_height=2,
    concat_attn_to_dec_input=True,
    model_mode=0,   # 0: train s2s; 1: train RL unc; 2: joint
    scheduled_sampling=.2,
    pondering_limit=3,
    uncertainty_sample_num=5,
    uncertainty_loss_weight=1.,
    reward_discount=.5,
)
UNC_S2S_CONF[DS_GEOQUERY_SP] = UNC_S2S_CONF[DS_GEOQUERY]
UNC_S2S_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=20,
    max_decoding_len=60,
    num_heads=2,
    num_enc_layers=2,
    encoder='lstm',
    residual_dropout=.1,
    attention_dropout=.1,
    feedforward_dropout=.1,
    intermediate_dropout=.5,
    vanilla_wiring=True,
    decoder='n_lstm',
    enc_attn="bilinear",
    dec_hist_attn="dot_product",
    dec_cell_height=2,
    concat_attn_to_dec_input=True,
    model_mode=0,   # 0: train s2s; 1: train RL unc; 2: joint
    scheduled_sampling=.2,
    pondering_limit=3,
    uncertainty_sample_num=5,
    uncertainty_loss_weight=1.,
    reward_discount=.5,
)

ST_ADA_TRANS2SEQ = 'ada_trans2s'
ADA_TRANS2SEQ_CONF = dict()
ADA_TRANS2SEQ_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=20,
    max_decoding_len=60,
    num_heads=8,
    num_enc_layers=2,
    encoder='lstm',
    decoder='lstm',
    act_max_layer=3,
    act=False,
    act_dropout=.3,
    act_epsilon=.1,
    residual_dropout=.1,
    attention_dropout=.1,
    feedforward_dropout=.1,
    embedding_dropout=.5,
    decoder_dropout=.5,
    prediction_dropout=.4,
    vanilla_wiring=False,
    act_loss_weight=-0.1,
    dwa="dot_product",
    enc_attn="dot_product",
    dec_hist_attn="dot_product",
    act_mode='mean_field',
    dec_cell_height=2,
)

ST_TRANSFORMER = 'transformer'
TRANSFORMER_CONF = dict()
TRANSFORMER_CONF[DS_GEOQUERY] = dict(
    emb_sz=256,
    batch_sz=32,
    max_decoding_len=60,
    num_heads=8,
    num_layers=6,
)
TRANSFORMER_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=32,
    max_decoding_len=70,
    num_heads=8,
    num_layers=2,
)

ST_UTRANSFORMER = 'ut'
UTRANSFORMER_CONF = dict()
UTRANSFORMER_CONF[DS_GEOQUERY] = dict(
    emb_sz=256,
    batch_sz=100,
    max_decoding_len=60,
    num_heads=8,
    max_num_layers=1,
    act=True,
)
UTRANSFORMER_CONF[DS_ATIS] = dict(
    emb_sz=256,
    batch_sz=100,
    max_decoding_len=70,
    num_heads=8,
    max_num_layers=1,
    act=True,
    residual_dropout=.05,
    attention_dropout=.001,
    feedforward_dropout=.05,
    vanilla_wiring=False,
)

SETTINGS = {
    ST_SEQ2SEQ: SEQ2SEQ_CONF,
    ST_UNC_S2S: UNC_S2S_CONF,
    ST_TRANS2SEQ: TRANS2SEQ_CONF,
    ST_TRANSFORMER: TRANSFORMER_CONF,
    ST_UTRANSFORMER: UTRANSFORMER_CONF,
    ST_ADA_TRANS2SEQ: ADA_TRANS2SEQ_CONF,
}

