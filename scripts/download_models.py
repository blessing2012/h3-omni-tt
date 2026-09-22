#!/usr/bin/env python3
"""Download the pinned public artifacts; no inference or GPU access."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm="sha256", git_blob=False):
    h = hashlib.new(algorithm)
    if git_blob:
        h.update(b"blob " + str(path.stat().st_size).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify(root, manifest, full_hash=False):
    base, lora = manifest["base"], manifest["lora"]
    for item in base["files"]:
        path = root / base["local_dir"] / item["path"]
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"Missing or wrong-size model file: {path}")
        if "git_blob_sha1" in item:
            if digest(path, "sha1", git_blob=True) != item["git_blob_sha1"]:
                raise ValueError(f"Base configuration/content mismatch: {path}")
        elif full_hash and digest(path) != item["sha256"]:
            raise ValueError(f"Base weight SHA-256 mismatch: {path}")
    path = root / lora["local_dir"] / lora["filename"]
    if not path.is_file() or path.stat().st_size != lora["size"] or digest(path) != lora["sha256"]:
        raise ValueError(f"LoRA SHA-256 mismatch: {path}")
    print(json.dumps({"verified": True, "base_files": len(base["files"]),
                      "base_large_weights_hashed": full_hash, "lora_hashed": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "weights")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--full-hash", action="store_true", help="Also read and hash every large base weight")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        for kind in ("base", "lora"):
            item = manifest[kind]
            patterns = ([x["path"] for x in item["files"]] if kind == "base" else [item["filename"]])
            snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                              local_dir=args.root / item["local_dir"],
                              allow_patterns=patterns + ["README.md", "LICENSE*", "NOTICE*"], max_workers=4)
    verify(args.root, manifest, full_hash=args.full_hash)


if __name__ == "__main__":
    main()
