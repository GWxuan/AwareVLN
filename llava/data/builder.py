from typing import Any

from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils.logging import logger

__all__ = ["build_dataset", "build_dataset_legacy"]


class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    datasets = []
    for name in mixture.strip().lower().split("+"):
        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1

        if name not in DATASETS_LEGACY:
            raise ValueError(f"Dataset {name} is not registered.")

        logger.warning(f"Dataset {name} is registered under legacy mode.")
        dataset = build_dataset_legacy(
            name,
            data_args=data_args,
            training_args=training_args,
            tokenizer=tokenizer,
        )

        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import LazyVLNCEDataset, LazyVLNCEWithReasoningDataset

    dataset_info = DATASETS_LEGACY[name]
    dataset_type = dataset_info.dataset_type
    if dataset_type == "vlnce":
        dataset_cls = LazyVLNCEDataset
    elif dataset_type == "vlnce_cot":
        dataset_cls = LazyVLNCEWithReasoningDataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    if dataset_type == "vlnce_cot":
        return dataset_cls(
            tokenizer=tokenizer,
            data_path=dataset_info.data_path,
            image_folder=dataset_info.image_path,
            data_args=data_args,
            training_args=training_args,
            reasoning_json_path=dataset_info.reason_path,
        )

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset_info.data_path,
        image_folder=dataset_info.image_path,
        data_args=data_args,
        training_args=training_args,
    )
