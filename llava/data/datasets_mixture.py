# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from dataclasses import dataclass, field

DATA_ROOT = os.environ.get("AWAREVLN_DATA_ROOT", "data")


@dataclass
class Dataset:
    dataset_name: str
    dataset_type: str
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})
    reason_path: str = field(default=None, metadata={"help": "Path to the reasoning data."})
    description: str = field(default=None, metadata={"help": "Dataset description."})


DATASETS_LEGACY = {}


def add_dataset(dataset: Dataset) -> None:
    if dataset.dataset_name in DATASETS_LEGACY:
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    DATASETS_LEGACY[dataset.dataset_name] = dataset


def register_datasets_mixtures() -> None:
    add_dataset(
        Dataset(
            dataset_name="r2r",
            dataset_type="vlnce_cot",
            data_path=os.path.join(DATA_ROOT, "reason/r2r/_anno_cot/annotations_shuffle_uni.json"),
            image_path=os.path.join(DATA_ROOT, "reason/r2r/videos"),
            reason_path=os.path.join(DATA_ROOT, "reason/r2r/_anno_cot/cot_new.json"),
            description="VLN-CE R2R training data with chain-of-thought reasoning.",
        )
    )

    add_dataset(
        Dataset(
            dataset_name="rxr",
            dataset_type="vlnce_cot",
            data_path=os.path.join(DATA_ROOT, "reason/rxr/_anno_cot/annotations_shuffle_uni.json"),
            image_path=os.path.join(DATA_ROOT, "reason/rxr/videos"),
            reason_path=os.path.join(DATA_ROOT, "reason/rxr/_anno_cot/cot_new.json"),
            description="RxR training data with chain-of-thought reasoning.",
        )
    )

    add_dataset(
        Dataset(
            dataset_name="r2rfollow",
            dataset_type="vlnce_cot",
            data_path=os.path.join(DATA_ROOT, "reason/r2rfollow/_anno_cot/annotations_shuffle_uni.json"),
            image_path=os.path.join(DATA_ROOT, "reason/r2rfollow/videos"),
            reason_path=os.path.join(DATA_ROOT, "reason/r2rfollow/_anno_cot/cot_new.json"),
            description="R2R follow-up training data.",
        )
    )

    add_dataset(
        Dataset(
            dataset_name="rxrfollow",
            dataset_type="vlnce_cot",
            data_path=os.path.join(DATA_ROOT, "reason/rxrfollow/_anno_cot/annotations_shuffle_uni.json"),
            image_path=os.path.join(DATA_ROOT, "reason/rxrfollow/videos"),
            reason_path=os.path.join(DATA_ROOT, "reason/rxrfollow/_anno_cot/cot_new.json"),
            description="RxR follow-up training data.",
        )
    )

    add_dataset(
        Dataset(
            dataset_name="human",
            dataset_type="vlnce",
            data_path=os.path.join(DATA_ROOT, "Human/annotations_shuffled.json"),
            image_path=os.path.join(DATA_ROOT, "Human/raw_frames"),
            description="Human demonstration navigation data.",
        )
    )
