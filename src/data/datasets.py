import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset


class MemoryDataset(Dataset):
    """Режим полного датасета в памяти."""

    def __init__(self, df: pd.DataFrame):
        self.X = (
            torch.tensor(df.drop(columns=["target"]).values, dtype=torch.float32) / 16.0
        )
        self.y = torch.tensor(df["target"].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StreamingDigitsDataset(IterableDataset):
    """Режим стриминга чанков с диска без полной загрузки файла в память."""

    def __init__(
        self,
        file_path: str,
        chunk_size: int,
        is_train: bool = True,
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.is_train = is_train
        self.train_ratio = train_ratio
        self.seed = seed

    def __iter__(self):
        chunks = pd.read_csv(self.file_path, chunksize=self.chunk_size)

        for chunk in chunks:
            train_chunk = chunk.sample(frac=self.train_ratio, random_state=self.seed)
            if not self.is_train:
                test_chunk = chunk.drop(train_chunk.index)
                current_chunk = test_chunk
            else:
                current_chunk = train_chunk

            X = current_chunk.drop(columns=["target"]).values / 16.0
            y = current_chunk["target"].values

            for i in range(len(X)):
                yield torch.tensor(X[i], dtype=torch.float32), torch.tensor(
                    y[i], dtype=torch.long
                )
