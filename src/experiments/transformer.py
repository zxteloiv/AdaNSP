import os.path
import datetime
import config
import utils.opt_parser
import data_adapter
import torch
import allennlp
import allennlp.data
import allennlp.modules
import allennlp.models
import allennlp.training
import allennlp.predictors

from allennlp.data.iterators import BucketIterator
from models.parallel_seq2seq import ParallelSeq2Seq
from models.transformer.encoder import TransformerEncoder
from models.transformer.decoder import TransformerDecoder
from allennlp.common.util import START_SYMBOL, END_SYMBOL

def main():
    parser = utils.opt_parser.get_trainer_opt_parser()
    parser.add_argument('models', nargs='*', help='pretrained models for the same setting')
    parser.add_argument('--test', action="store_true", help='use testing mode')
    parser.add_argument('--num-layer', type=int, help="stacked layer of transformer model")

    args = parser.parse_args()

    reader = data_adapter.GeoQueryDatasetReader()
    training_set = reader.read(config.DATASETS[args.dataset].train_path)
    try:
        validation_set = reader.read(config.DATASETS[args.dataset].dev_path)
    except:
        validation_set = None

    vocab = allennlp.data.Vocabulary.from_instances(training_set)
    st_ds_conf = config.TRANSFORMER_CONF[args.dataset]
    if args.num_layer:
        st_ds_conf['num_layers'] = args.num_layer

    encoder = TransformerEncoder(input_dim=st_ds_conf['emb_sz'],
                                 num_layers=st_ds_conf['num_layers'],
                                 num_heads=st_ds_conf['num_heads'],
                                 feedforward_hidden_dim=st_ds_conf['emb_sz'],)
    decoder = TransformerDecoder(input_dim=st_ds_conf['emb_sz'],
                                 num_layers=st_ds_conf['num_layers'],
                                 num_heads=st_ds_conf['num_heads'],
                                 feedforward_hidden_dim=st_ds_conf['emb_sz'],
                                 feedforward_dropout=0.1,)
    source_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('nltokens'),
                                                  embedding_dim=st_ds_conf['emb_sz'])
    target_embedding = allennlp.modules.Embedding(num_embeddings=vocab.get_vocab_size('lftokens'),
                                                  embedding_dim=st_ds_conf['emb_sz'])
    model = ParallelSeq2Seq(vocab=vocab,
                            encoder=encoder,
                            decoder=decoder,
                            source_embedding=source_embedding,
                            target_embedding=target_embedding,
                            target_namespace='lftokens',
                            start_symbol=START_SYMBOL,
                            eos_symbol=END_SYMBOL,
                            max_decoding_step=st_ds_conf['max_decoding_len'],
                            )

    if args.models:
        model.load_state_dict(torch.load(args.models[0]))

    if not args.test or not args.models:
        iterator = BucketIterator(sorting_keys=[("source_tokens", "num_tokens")], batch_size=st_ds_conf['batch_sz'])
        iterator.index_with(vocab)

        optim = torch.optim.Adam(model.parameters())

        savepath = os.path.join(config.SNAPSHOT_PATH, args.dataset, 'transformer',
                                datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
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

        for instance in testing_set:
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