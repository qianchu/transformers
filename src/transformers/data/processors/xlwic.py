# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" XNLI utils (dataset loading and evaluation) """


import logging
import os

from .utils import DataProcessor, InputExample


logger = logging.getLogger(__name__)


class XlwicProcessor(DataProcessor):
    """Processor for the XNLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self, language, train_language=None):
        self.language = language
        self.train_language = train_language

    def get_train_examples(self, data_dir):
        """See base class."""
        lg = self.language if self.train_language is None else self.train_language
        fname=os.path.join(data_dir, "/{}/train.tsv".format(lg))
        logger.info("train file: "+fname)
        lines = self._read_tsv(fname)
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % ("train", i)
            text_a = line[0]
            text_b = line[1]
            label = "T" if line[2] == "T" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_test_examples(self, data_dir,testset='test'):
        """See base class."""
        lg = self.language
        lines = self._read_tsv(os.path.join(data_dir, "{0}/{1}.tsv".format(lg,testset)))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (testset, i)
            text_a = line[0]
            text_b = line[1]
            label = "T" if line[2] == "T" else line[2]
            assert isinstance(text_a, str) and isinstance(text_b, str) and isinstance(label, str)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

   

    def get_labels(self):
        """See base class."""
        return ["T","F"]


xlwic_processors = {
    "xlwic": XlwicProcessor,
}

xlwic_output_modes = {
    "xlwic": "classification",
}

xlwic_tasks_num_labels = {
    "xlwic": 2,
}
