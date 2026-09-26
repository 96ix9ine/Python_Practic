import os
import json
import hashlib
import sys
import pandas as pd
from sklearn.datasets import load_digits
from configs.config_schema import Lab1Config


def calculate_sha256(filepath: str) -> str:
    """Вычисление SHA-256 хеша файла порциями."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    print("STATUS: Запуск процесса скачивания и верификации данных...")

    try:
        cfg = Lab1Config()
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs(cfg.eda.output_dir, exist_ok=True)

        print("STATUS: Извлечение датасета load_digits из scikit-learn...")
        digits = load_digits(n_class=cfg.data.n_classes)
        df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
        df["target"] = digits.target

        raw_path = "data/raw/digits.csv"
        df.to_csv(raw_path, index=False)

        file_size = os.path.getsize(raw_path)
        actual_hash = calculate_sha256(raw_path)

        expected_hash = cfg.data.expected_hash
        if actual_hash != expected_hash:
            print(
                f"CRITICAL ERROR: Критческая ошибка верификации! Хеш файла не совпадает с эталоном."
            )
            print(f"Ожидался: {expected_hash}")
            print(f"Получен:  {actual_hash}")
            sys.exit(1)

        manifest = {
            raw_path: {
                "hash": actual_hash,
                "size_bytes": file_size,
                "description": "Verified raw data from sklearn.datasets.load_digits",
            }
        }

        manifest_path = os.path.join(cfg.eda.output_dir, cfg.eda.manifest_filename)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

        print(f"SUCCESS: Данные успешно верифицированы и сохранены. Хеш: {actual_hash}")
        sys.exit(0)

    except Exception as e:
        print(f"CRITICAL ERROR: Произошел сбой при обработке данных: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
