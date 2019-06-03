from typing import Dict, Mapping, Iterator, List

import os.path
import datetime
import torch
import tqdm

import allennlp.data
from allennlp.data.iterators import BucketIterator

import allennlp.training

import allennlp
import allennlp.common
import allennlp.models
import allennlp.modules
import allennlp.predictors
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding

import config

import data_adapter
import utils.opt_parser
from models.transformer.encoder import TransformerEncoder, UTEncoder


def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--num-layer', type=int, help='maximum number of stacked layers')
    parser.add_argument('--use-ut', action="store_true", help='Use universal transformer instead of transformer')

    args = parser.parse_args()

    reader = data_adapter.GeoQueryDatasetReader()
    training_set = reader.read(config.DATASETS[args.dataset].train_path)
    try:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    st_ds_conf = config.TRANS2SEQ_CONF[args.dataset]
    if args.num_layer:
        st_ds_conf['max_num_layers'] = args.num_layer
    if args.epoch:
        config.TRAINING_LIMIT = args.epoch
    if args.batch:
        st_ds_conf['batch_sz'] = args.batch
    bsz = st_ds_conf['batch_sz']
    emb_sz = st_ds_conf['emb_sz']

    src_embedder = BasicTextFieldEmbedder(
        token_embedders={ "tokens": Embedding(vocab.get_vocab_size('nltokens'), emb_sz)}
    )

    if args.use_ut:
        transformer_encoder = UTEncoder(input_dim=emb_sz,
                                        max_num_layers=st_ds_conf['max_num_layers'],
                                        num_heads=st_ds_conf['num_heads'],
                                        feedforward_hidden_dim=emb_sz,
                                        feedforward_dropout=st_ds_conf['feedforward_dropout'],
                                        attention_dropout=st_ds_conf['attention_dropout'],
                                        residual_dropout=st_ds_conf['residual_dropout'],
                                        use_act=st_ds_conf['act'],
                                        use_vanilla_wiring=st_ds_conf['vanilla_wiring'])
    else:
        transformer_encoder = TransformerEncoder(input_dim=emb_sz,
                                                 num_layers=st_ds_conf['max_num_layers'],
                                                 num_heads=st_ds_conf['num_heads'],
                                                 feedforward_hidden_dim=emb_sz,
                                                 feedforward_dropout=st_ds_conf['feedforward_dropout'],
                                                 attention_dropout=st_ds_conf['attention_dropout'],
                                                 residual_dropout=st_ds_conf['residual_dropout'],
                                                 )

    model = allennlp.models.SimpleSeq2Seq(
        vocab,
        source_embedder=src_embedder,
        encoder=transformer_encoder,
        max_decoding_steps=50,
        attention=allennlp.modules.attention.DotProductAttention(),
        beam_size=6,
        target_namespace="lftokens",
        use_bleu=True
    )

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=bsz)
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters())

        savepath = os.path.join(config.SNAPSHOT_PATH, args.dataset, 'transformer2seq',
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
            cuda_device=args.device,
            num_epochs=config.TRAINING_LIMIT,
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        predictor = allennlp.predictors.SimpleSeq2SeqPredictor(model, reader)

        for instance in tqdm.tqdm(testing_set, total=len(testing_set)):
            print('SRC: ', instance.fields['source_tokens'].tokens)
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            del instance.fields['target_tokens']
            output = predictor.predict_instance(instance)
            print('PRED:', ' '.join(output['predicted_tokens']))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass


