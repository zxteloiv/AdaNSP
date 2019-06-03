from typing import Dict, Mapping, Iterator, List

import os.path
import datetime
import torch

import allennlp.data
from allennlp.data.iterators import BucketIterator

import allennlp.training

import allennlp
import allennlp.common
import allennlp.models
import allennlp.modules
import allennlp.predictors
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import TokenEmbedder, Embedding

import config

import data_adapter
import utils.opt_parser


def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--enc-layers', type=int, default=1, help="encoder layer number defaulted to 1")
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--use-dev', action="store_true")

    args = parser.parse_args()

    reader = data_adapter.GeoQueryDatasetReader()
    training_set = reader.read(config.DATASETS[args.dataset].train_path)
    if args.use_dev:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    st_ds_conf = config.SEQ2SEQ_CONF[args.dataset]
    if args.epoch:
        config.TRAINING_LIMIT = args.epoch
    bsz = st_ds_conf['batch_sz']
    emb_sz = st_ds_conf['emb_sz']

    src_embedder = BasicTextFieldEmbedder(
        token_embedders={ "tokens": Embedding(vocab.get_vocab_size('nltokens'), emb_sz)}
    )

    encoder = PytorchSeq2SeqWrapper(torch.nn.LSTM(emb_sz, emb_sz,
                                                  num_layers=args.enc_layers, batch_first=True))

    model = allennlp.models.SimpleSeq2Seq(
        vocab,
        source_embedder=src_embedder,
        encoder=encoder,
        max_decoding_steps=st_ds_conf['max_decoding_len'],
        attention=allennlp.modules.attention.DotProductAttention(),
        beam_size=8,
        target_namespace="lftokens",
        use_bleu=True
    )

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=bsz)
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters())

        savepath = os.path.join(config.SNAPSHOT_PATH, args.dataset, 'seq2seq',
                                datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + "--" + args.memo)
        if not os.path.exists(savepath):
            os.makedirs(savepath, mode=0o755)

        trainer = allennlp.training.Trainer(
            model=model,
            optimizer=optim,
            iterator=iterator,
            train_dataset=training_set,
            validation_dataset=validation_set if args.use_dev else None,
            serialization_dir=savepath,
            cuda_device=args.device,
            num_epochs=config.TRAINING_LIMIT,
        )

        trainer.train()

    else:
        testing_set = reader.read(config.DATASETS[args.dataset].test_path)
        model.eval()

        predictor = allennlp.predictors.SimpleSeq2SeqPredictor(model, reader)

        for instance in testing_set:
            print('SRC: ', instance.fields['source_tokens'].tokens)
            print('GOLD:', ' '.join(str(x) for x in instance.fields['target_tokens'].tokens[1:-1]))
            print('PRED:', ' '.join(predictor.predict_instance(instance)['predicted_tokens']))


if __name__ == '__main__':
    main()


