# Copyright 2023 The Alpaca Team
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

"""Utils for hosting Alpaca eval on spaces."""

import json

import datasets

_CITATION = """
@misc{alpaca_eval,
  author = {Xuechen Li and Tianyi Zhang and Yann Dubois and Rohan Taori and Ishaan Gulrajani and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {AlpacaEval: An Automatic Evaluator of Instruction-following Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\\url{https://github.com/tatsu-lab/alpaca_eval}}
}
"""

# You can copy an official description
_DESCRIPTION = """
Data for alpaca_eval, which aims to help automatic evaluation of instruction-following models
"""

_HOMEPAGE = "https://huggingface.co/datasets/tatsu-lab/alpaca_eval"

_LICENSE = "CC BY 4.0"

# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    "alpaca_eval": "alpaca_eval.json",
    "alpaca_eval_gpt4_baseline": "alpaca_eval_gpt4_baseline.json",
    "alpaca_eval_all_outputs": "alpaca_eval_all_outputs.json",
    "alpaca_farm_human_annotations": "alpaca_farm_human_annotations.json",
    "alpaca_farm_human_crossannotations": "alpaca_farm_human_crossannotations.json",
    "alpaca_eval_annotations_alpaca_eval_gpt4": "alpaca_eval_annotations_alpaca_eval_gpt4.json",
    "alpaca_eval_annotations_claude": "alpaca_eval_annotations_claude.json",
}


class AlpacaFarmDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="alpaca_eval", version=VERSION, description="Official AlpacaEval 1.0 evaluation set."
        ),
        datasets.BuilderConfig(
            name="alpaca_eval_gpt4_baseline", version=VERSION, description="Official AlpacaEval 2.0 evaluation set."
        ),
        datasets.BuilderConfig(
            name="alpaca_eval_all_outputs", version=VERSION, description="Outputs from the AlpacaEval leaderboard."
        ),
        datasets.BuilderConfig(
            name="alpaca_farm_human_annotations",
            version=VERSION,
            description="Human annotations of 21 models on the AlpacaFarm evaluation",
        ),
        datasets.BuilderConfig(
            name="alpaca_farm_human_crossannotations",
            version=VERSION,
            description="650 pairs cross-annotated by 4 humans.",
        ),
        datasets.BuilderConfig(
            name="alpaca_eval_annotations_alpaca_eval_gpt4",
            version=VERSION,
            description="Leaderboard annotations by alpaca_eval_gpt4.",
        ),
        datasets.BuilderConfig(
            name="alpaca_eval_annotations_claude",
            version=VERSION,
            description="Leaderboard annotations by Claude.",
        )
    ]

    DEFAULT_CONFIG_NAME = "alpaca_eval"

    def _info(self):
        if self.config.name in ("alpaca_eval","alpaca_eval_gpt4_baseline","alpaca_eval_all_outputs"):
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "output": datasets.Value("string"),
                    "generator": datasets.Value("string"),
                    "dataset": datasets.Value("string"),
                }
            )
        elif self.config.name in ("alpaca_farm_human_annotations"):
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "output_1": datasets.Value("string"),
                    "output_2": datasets.Value("string"),
                    "preference": datasets.Value("int64"),
                    "annotator_index": datasets.Value("int64"),
                    "dataset": datasets.Value("string"),
                    "datasplit": datasets.Value("string"),
                    "generator": datasets.Value("string"),
                    "sample_mode": datasets.Value("string"),
                }
            )
        elif self.config.name in ( "alpaca_farm_human_crossannotations"):
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "output_1": datasets.Value("string"),
                    "output_2": datasets.Value("string"),
                    "preference": datasets.Value("int64"),
                    "annotator_index": datasets.Value("int64"),
                    "dataset": datasets.Value("string"),
                    "datasplit": datasets.Value("string"),
                }
            )
        elif self.config.name in ( "alpaca_eval_annotations_claude", "alpaca_eval_annotations_alpaca_eval_gpt4"):
            features = datasets.Features(
                {
                    "instruction": datasets.Value("string"),
                    "output_1": datasets.Value("string"),
                    "generator_1": datasets.Value("string"),
                    "output_2": datasets.Value("string"),
                    "generator_2": datasets.Value("string"),
                    "dataset": datasets.Value("string"),
                    "annotator": datasets.Value("string"),
                    "preference": datasets.Value("int64"),
                    "price_per_example": datasets.Value("float"),
                    "time_per_example": datasets.Value("float")
                }
            )
        else:
            raise NotImplementedError()

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls = _URLS[self.config.name]
        data_dir = dl_manager.download_and_extract(urls)
        print(data_dir)
        if self.config.name in ("alpaca_eval","alpaca_eval_gpt4_baseline", "alpaca_eval_all_outputs", "alpaca_eval_annotations_claude", "alpaca_eval_annotations_alpaca_eval_gpt4"):
            return [
                datasets.SplitGenerator(
                    name="eval",
                    gen_kwargs={
                        "filepath": data_dir,
                        "split": "eval",
                    },
                )
            ]
        elif self.config.name in ("alpaca_farm_human_annotations", "alpaca_farm_human_crossannotations"):
            return [
                datasets.SplitGenerator(
                    name="validation",
                    gen_kwargs={
                        "filepath": data_dir,
                        "split": "validation",
                    },
                )
            ]

    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        for key, example in enumerate(data):
            if self.config.name in (
                "alpaca_eval",
                "alpaca_eval_gpt4_baseline",
                "alpaca_eval_all_outputs",
                "alpaca_farm_human_annotations",
                "alpaca_farm_human_crossannotations",
                "alpaca_eval_annotations_claude",
                "alpaca_eval_annotations_alpaca_eval_gpt4",
            ):
                # Yields examples as (key, example) tuples
                yield key, example
            else:
                raise NotImplementedError()