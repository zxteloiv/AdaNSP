from typing import Iterable, Dict, List, Tuple
from utils.file_reader import open_json

import allennlp.data.dataset_readers

from allennlp.data import Instance, Token
from allennlp.data.fields import TextField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.common.util import START_SYMBOL, END_SYMBOL


class GeoQueryDatasetReader(allennlp.data.DatasetReader):
    def __init__(self, lazy=False):
        super(GeoQueryDatasetReader, self).__init__(lazy)

        self.instance_keys = ("question", "logic_form")
        self.instance_keys = ("source_tokens", "target_tokens")

    def _read(self, file_path: str) -> Iterable[Instance]:
        for json_obj in open_json(file_path):
            src, tgt = json_obj['src'], json_obj['tgt']
            instance = self.text_to_instance(src, tgt)

            yield instance

    def text_to_instance(self, question: List[str], logic_form: List[str]) -> Instance:

        x = TextField(list(map(Token, question)),
                      {'tokens': SingleIdTokenIndexer('nltokens')})
        z = TextField(list(map(Token, [START_SYMBOL] + logic_form + [END_SYMBOL])),
                      {'tokens': SingleIdTokenIndexer('lftokens')})
        instance = Instance(dict(zip(self.instance_keys, (x, z))))

        return instance


