# encoding: utf-8

# author: BrikerMan
# contact: eliyar917@gmail.com
# blog: https://eliyar.biz

# file: test_load_and_save.py
# time: 1:51 下午

import os

os.environ['TF_KERAS'] = '1'
import unittest

from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import to_array


class TestBaseUse(unittest.TestCase):
    def test_load_and_save(self):
        current_folder = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
        bert_path = os.path.join(current_folder, 'assets', 'bert_sample_model')

        config_path = os.path.join(bert_path, 'bert_config.json')
        checkpoint_path = os.path.join(bert_path, 'bert_model.ckpt')
        dict_path = os.path.join(bert_path, 'vocab.txt')
        bert_model = build_transformer_model(config_path=config_path,
                                             checkpoint_path=checkpoint_path,
                                             model='bert',
                                             application='encoder',
                                             return_keras_model=True)

        tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

        # 编码测试
        token_ids, segment_ids = tokenizer.encode(u'jack play all day')
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        print('\n ===== predicting =====\n')
        print(bert_model.predict([token_ids, segment_ids]))

        # Serialize model
        _ = bert_model.to_json()


if __name__ == "__main__":
    pass
