import os
import json
import hashlib
import pandas as pd
from sklearn.datasets import load_digits
from configs.config_schema import Lab1Config


def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    cfg = Lab1Config()
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(cfg.eda.output_dir, exist_ok=True)

    # Загрузка
    digits = load_digits(n_class=cfg.data.n_classes)
    df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
    df["target"] = digits.target

    # Сохранение сырого датасета
    raw_path = "data/raw/digits.csv"
    df.to_csv(raw_path, index=False)

    # Расчет хеша и размера
    file_size = os.path.getsize(raw_path)
    file_hash = calculate_sha256(raw_path)

    manifest = {
        raw_path: {
            "hash": file_hash,
            "size_bytes": file_size,
            "description": "Raw data from sklearn.datasets.load_digits",
        }
    }

    manifest_path = os.path.join(cfg.eda.output_dir, cfg.eda.manifest_filename)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"Dataset saved to {raw_path}. Hash: {file_hash}")


if __name__ == "__main__":
    main()
