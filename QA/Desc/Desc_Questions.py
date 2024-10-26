# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
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

#Lint as: python3
"""Description and Questions Dataset"""


import json

import datasets
from datasets.tasks import QuestionAnsweringExtractive
import pandas as pd


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@article{2016arXiv160605250R,
       author = {{Rajpurkar}, Pranav and {Zhang}, Jian and {Lopyrev},
                 Konstantin and {Liang}, Percy},
        title = "{SQuAD: 100,000+ Questions for Machine Comprehension of Text}",
      journal = {arXiv e-prints},
         year = 2016,
          eid = {arXiv:1606.05250},
        pages = {arXiv:1606.05250},
archivePrefix = {arXiv},
       eprint = {1606.05250},
}
"""

_DESCRIPTION = """\
Image descriptions for data science charts
"""

_URL = "https://huggingface.co/datasets/eduvedras/Desc_Questions/resolve/main/images.tar.gz"

class Desc_QuestionsTargz(datasets.GeneratorBasedBuilder):

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "Chart": datasets.Image(),
                    "Description": datasets.Value("string"),
                    "Chart_name": datasets.Value("string"),
                    "Questions": datasets.Value("string"),
                }
            ),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
            homepage="https://huggingface.co/datasets/eduvedras/Desc_Questions",
            citation=_CITATION,
            task_templates=[
                QuestionAnsweringExtractive(
                    question_column="question", context_column="context", answers_column="answers"
                )
            ],
        )

    def _split_generators(self, dl_manager):
        path = dl_manager.download(_URL)
        image_iters = dl_manager.iter_archive(path)
        metadata_train_path = "https://huggingface.co/datasets/eduvedras/Desc_Questions/resolve/main/desc_questions_dataset_train1.csv" 
        metadata_test_path = "https://huggingface.co/datasets/eduvedras/Desc_Questions/resolve/main/desc_questions_dataset_test1.csv"

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"images": image_iters,
                                                                           "metadata_path": metadata_train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"images": image_iters,
                                                                           "metadata_path": metadata_test_path}),
        ]

    def _generate_examples(self, images, metadata_path):
        metadata = pd.read_csv(metadata_path, sep=';')
        idx = 0
        for index, row in metadata.iterrows():
            for filepath, image in images:
                filepath = filepath.split('/')[-1]
                if row['Chart'] in filepath:
                    yield idx, {
                        "Chart": {"path": filepath, "bytes": image.read()},
                        "Description": row['description'],
                        "Chart_name": row['Chart'],
                        "Questions": row['Questions'],
                    }
                    break
            idx += 1