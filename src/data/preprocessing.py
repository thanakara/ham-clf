import logging

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split


class HAMPreprocessor:
    def __init__(self, filepath: Path | str, log: logging.Logger):
        self.filepath = filepath
        self.log = log
        self.train_size_split = 0.7
        self.target_samples_per_class = {
            "melanocytic_nevi": 2500,
            "melanoma": 1113,
            "benign_keratosis": 1100,
            "basal_cell_carcinoma": 900,
            "actinic_keratoses": 600,
            "vascular_lesions": 300,
            "dermatofibroma": 300,
        }

    @staticmethod
    def _balance_classes(df: pd.DataFrame, target_samples_per_class: dict[str, int]) -> pd.DataFrame:
        balanced = []

        for class_name, target_count in target_samples_per_class.items():
            class_df = df[df.dx == class_name]
            current_count = len(class_df)

            if current_count >= target_count:
                sampled = class_df.sample(target_count, random_state=42)
            else:
                sampled = class_df.sample(target_count, replace=True, random_state=42)

            balanced.append(sampled)

        return pd.concat(balanced).sample(frac=1.0, random_state=42).reset_index(drop=True)

    @staticmethod
    def _get_one_image_per_lesion(df: pd.DataFrame) -> pd.DataFrame:
        return df.groupby("lesion_id", group_keys=False).apply(lambda X: X.sample(1, random_state=42))

    def _load_df(self) -> pd.DataFrame:
        self.log.info("loading_dataframe")
        usecols = ["image_id", "dx", "lesion_id"]
        img_suffix = ".jpg"
        df = pd.read_csv(self.filepath, usecols=usecols)
        dx_names_short = {
            "akiec": "actinic_keratoses",
            "bcc": "basal_cell_carcinoma",
            "bkl": "benign_keratosis",
            "df": "dermatofibroma",
            "mel": "melanoma",
            "nv": "melanocytic_nevi",
            "vasc": "vascular_lesions",
        }
        df.dx = df.dx.map(dx_names_short).astype("category")
        df.image_id = df.image_id + img_suffix
        return df

    def _get_unique_lesions(self) -> tuple[pd.DataFrame]:
        df = self._load_df()
        unique_lesions = df[["lesion_id", "dx"]].drop_duplicates()
        test_size = 1.0 - self.train_size_split
        train_lesions, temp_lesions = train_test_split(
            unique_lesions,
            test_size=test_size,
            stratify=unique_lesions.dx,
            random_state=42,
        )
        valid_lesions, test_lesions = train_test_split(
            temp_lesions, test_size=0.5, stratify=temp_lesions.dx, random_state=42
        )
        train_df = df[df.lesion_id.isin(train_lesions.lesion_id)]
        valid_df = df[df.lesion_id.isin(valid_lesions.lesion_id)]
        test_df = df[df.lesion_id.isin(test_lesions.lesion_id)]
        self.log.info("dropping_duplicates_on_lesion")

        return train_df, valid_df, test_df

    def preprocess(self) -> tuple[pd.DataFrame]:
        train_df, valid_df, test_df = self._get_unique_lesions()
        self.log.info("balance_training_dataset")
        train = self._balance_classes(train_df, self.target_samples_per_class)
        self.log.info("get_one_image_per_lesion_on_validation_dataset")
        valid = self._get_one_image_per_lesion(valid_df)
        self.log.info("get_one_image_per_lesion_on_testing_dataset")
        test = self._get_one_image_per_lesion(test_df)
        return train, valid, test
