#!/usr/bin/env python3
"""Convert a Hugging Face dataset to a CSV file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and export it as CSV."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help='Hugging Face dataset path, e.g. "imdb" or "username/dataset".',
    )
    parser.add_argument(
        "--config",
        default=None,
        help='Optional dataset config/subset, e.g. "plain_text".',
    )
    parser.add_argument(
        "--split",
        default=None,
        help='Optional split name, e.g. "train". If omitted, all splits are combined.',
    )
    parser.add_argument(
        "--outpath",
        required=True,
        help=(
            "Output CSV path or template. Supported placeholders: "
            "{language}, {split}, {dataset}, {dataset_name}, {config}."
        ),
    )
    parser.add_argument(
        "--language",
        default="unknown",
        help="Language value used by the {language} outpath placeholder.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="Optional list of columns to keep.",
    )
    parser.add_argument(
        "--auto-split",
        action="store_true",
        help=(
            "If dataset has no predefined splits, create train/test splits "
            "from the loaded rows."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio used with --auto-split. Test ratio is (1 - train_ratio).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for --auto-split.",
    )
    return parser.parse_args()


def resolve_output_path(
    outpath_template: str,
    language: str,
    split: str,
    dataset: str,
    config: str | None,
) -> Path:
    values = {
        "language": language,
        "split": split,
        "dataset": dataset,
        "dataset_name": dataset.split("/")[-1],
        "config": config or "default",
    }
    try:
        rendered = outpath_template.format(**values)
    except KeyError as exc:
        supported = ", ".join(sorted(values))
        raise ValueError(
            f"Unknown outpath placeholder: {exc}. Supported placeholders: {supported}"
        ) from exc
    return Path(rendered).expanduser().resolve()


def keep_columns(df: pd.DataFrame, columns: list[str] | None) -> pd.DataFrame:
    if not columns:
        return df
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"These columns were not found in the dataset: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[columns]


def write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(df)} rows to {output_path}")


def to_dataframe(dataset_obj: Dataset | DatasetDict, split: str | None) -> pd.DataFrame:
    if isinstance(dataset_obj, Dataset):
        return dataset_obj.to_pandas()

    if split:
        if split not in dataset_obj:
            raise ValueError(
                f'Split "{split}" not found. Available splits: {list(dataset_obj.keys())}'
            )
        return dataset_obj[split].to_pandas()

    frames = []
    for split_name, split_ds in dataset_obj.items():
        frame = split_ds.to_pandas()
        frame.insert(0, "__split__", split_name)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()

    if not 0 < args.train_ratio < 1:
        raise ValueError("--train-ratio must be between 0 and 1 (exclusive).")

    dataset_obj = load_dataset(path=args.dataset, name=args.config)

    if isinstance(dataset_obj, Dataset) and args.auto_split:
        full_df = keep_columns(dataset_obj.to_pandas(), args.columns)
        train_df = full_df.sample(frac=args.train_ratio, random_state=args.seed)
        test_df = full_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

        train_path = resolve_output_path(
            outpath_template=args.outpath,
            language=args.language,
            split="train",
            dataset=args.dataset,
            config=args.config,
        )
        test_path = resolve_output_path(
            outpath_template=args.outpath,
            language=args.language,
            split="test",
            dataset=args.dataset,
            config=args.config,
        )
        write_csv(train_df, train_path)
        write_csv(test_df, test_path)
        return

    if isinstance(dataset_obj, DatasetDict) and args.split is None and "{split}" in args.outpath:
        for split_name, split_ds in dataset_obj.items():
            df = keep_columns(split_ds.to_pandas(), args.columns)
            output_path = resolve_output_path(
                outpath_template=args.outpath,
                language=args.language,
                split=split_name,
                dataset=args.dataset,
                config=args.config,
            )
            write_csv(df, output_path)
        return

    df = keep_columns(to_dataframe(dataset_obj, args.split), args.columns)
    output_path = resolve_output_path(
        outpath_template=args.outpath,
        language=args.language,
        split=args.split or "all",
        dataset=args.dataset,
        config=args.config,
    )
    write_csv(df, output_path)


if __name__ == "__main__":
    main()



#hausa:
# https://github.com/AsiyaZanga/HausaMovieReview

#Swahili:
# https://huggingface.co/datasets/michsethowusu/swahili-sentiments-corpus

#Igbo:
# https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter

# python3 ..//Deception_Detection/languages/extract_dataset.py \
#   --dataset "your_org/your_dataset" \
#   --language "hausa" \
#   --outpath "..//Deception_Detection/languages/{language}/raw/real_{language}_reviews.csv"

#If there is no predefined split in the dataset, you can use the --auto-split option to create train/test splits from the loaded rows. The following command will create a train/test split with 80% of the data in the train set and 20% in the test set, using a random seed of 42 for reproducibility.
# python3 ..//Deception_Detection/languages/extract_dataset.py \
#   --dataset "your_org/your_dataset" \
#   --language "korean" \
#   --auto-split \
#   --train-ratio 0.8 \
#   --seed 42 \
#   --outpath "..//Deception_Detection/languages/{language}/raw/{split}/real_{language}_{split}_reviews.csv"