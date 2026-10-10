#!/usr/bin/env python3
"""Manual, one-model Hugging Face backup. No secrets are printed or committed."""
import os
import shutil
import tempfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_url

SOURCE = "microsoft/TRELLIS.2-4B"
DESTINATION = "Altaqawy/TRELLIS.2-4B-backup"
MODE = os.environ.get("BACKUP_MODE", "audit")
TOKEN = os.environ.get("HF_TOKEN") if MODE == "backup" else None


def files_by_path(info):
    result = {}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if name:
            result[name] = sibling.size
    return result


def main():
    if MODE not in ("audit", "backup"):
        raise ValueError("BACKUP_MODE must be audit or backup")

    api = HfApi(token=TOKEN)
    source = api.model_info(SOURCE, files_metadata=True)
    license_id = str(getattr(source.card_data, "license", "") or "").lower()
    if license_id != "mit":
        raise RuntimeError(f"Expected MIT license for {SOURCE}; found {license_id!r}. Stop for review.")

    original = files_by_path(source)
    missing_sizes = [name for name, size in original.items() if size is None]
    if missing_sizes:
        raise RuntimeError(f"Source file sizes unavailable for: {missing_sizes[:5]}")
    total = sum(original.values())
    print(f"Source: {SOURCE} @ {source.sha}")
    print(f"Destination: {DESTINATION} (private)")
    print(f"Files: {len(original)} | Total: {total / (1024**3):.2f} GiB")
    if MODE == "audit":
        for name, size in sorted(original.items()):
            print(f"  {size / (1024**2):9.1f} MiB  {name}")
        print("AUDIT ONLY: no repository created or files copied.")
        return

    if not TOKEN:
        raise RuntimeError("Missing HF_TOKEN GitHub Actions secret.")
    api.create_repo(repo_id=DESTINATION, repo_type="model", private=True, exist_ok=True)
    dest = api.model_info(DESTINATION, files_metadata=True)
    if not dest.private:
        raise RuntimeError("Destination is public! Refusing to copy model files.")

    existing = files_by_path(dest)
    copied = 0
    skipped = 0

    with tempfile.TemporaryDirectory(prefix="hf-backup-") as folder:
        root = Path(folder)
        session = requests.Session()
        for name, size in sorted(original.items(), key=lambda pair: pair[1]):
            if existing.get(name) == size:
                skipped += 1
                print(f"SKIP (already present): {name}")
                continue

            available = shutil.disk_usage(root).free
            if size + (1024**3) > available:
                raise RuntimeError(
                    f"Insufficient runner disk for {name} ({size} bytes). "
                    "Use a larger runner or a different transfer method."
                )

            local = root / name
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {name} ({size / (1024**3):.2f} GiB)")
            url = hf_hub_url(repo_id=SOURCE, filename=name, revision=source.sha)
            with session.get(url, stream=True, timeout=(30, 240)) as response:
                response.raise_for_status()
                with local.open("wb") as file:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            file.write(chunk)
            if local.stat().st_size != size:
                raise RuntimeError(f"Downloaded size mismatch for {name}")

            print(f"Uploading {name} to private repository")
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=name,
                repo_id=DESTINATION,
                repo_type="model",
                commit_message=f"Back up {name} from {SOURCE} ({source.sha[:12]})",
            )
            local.unlink()
            copied += 1

    final = files_by_path(api.model_info(DESTINATION, files_metadata=True))
    mismatch = [
        name for name, size in original.items()
        if final.get(name) != size
    ]
    if mismatch:
        raise RuntimeError(f"Backup incomplete. Files missing/different size: {mismatch[:10]}")

    print(f"VERIFIED: {copied} uploaded, {skipped} already present.")
    print(f"Private backup: https://huggingface.co/{DESTINATION}")


if __name__ == "__main__":
    main()
