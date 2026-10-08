import os
import json
import hashlib


def calculate_sha256(filepath: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def main():
    print("STATUS: Сканирование репозитория и сборка манифеста артефактов ЛР8...")
    manifest_path = "reports/LAB8/artifacts_manifest.json"
    os.makedirs("reports/LAB8", exist_ok=True)

    manifest = {"artifacts": []}

    target_dirs = ["reports", "configs", "src"]

    for target_dir in target_dirs:
        if not os.path.exists(target_dir):
            continue
        for root, _, files in os.walk(target_dir):
            for file in files:
                if ".pytest_cache" in root or "__pycache__" in root or ".git" in root:
                    continue
                filepath = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(filepath)
                    file_hash = calculate_sha256(filepath)

                    manifest["artifacts"].append(
                        {
                            "file_path": filepath.replace("\\", "/"),
                            "size_bytes": file_size,
                            "sha256_hash": file_hash,
                        }
                    )
                except Exception:
                    continue

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=4, ensure_ascii=False)

    print(
        f"SUCCESS: Общий манифест файлов успешно сохранен со всеми хешами: {manifest_path}"
    )


if __name__ == "__main__":
    main()
