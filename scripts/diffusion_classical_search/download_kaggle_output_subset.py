#!/usr/bin/env python3
"""Download an allowlisted prefix from Kaggle kernel output without cache trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import time

import requests
from requests import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="dcs-results/")
    parser.add_argument("--page-size", type=int, default=200)
    parser.add_argument("--page-delay-seconds", type=float, default=1.0)
    parser.add_argument("--expected-count", type=int)
    return parser.parse_args()


def main() -> int:
    cli = parse_args()
    kernel_id = json.loads(cli.metadata.read_text(encoding="utf-8"))["id"]
    api = KaggleApi()
    owner, slug, _ = api.parse_kernel_string(kernel_id)
    cli.output.mkdir(parents=True, exist_ok=True)
    state_path = cli.output / "subset_download_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        token = state["next_page_token"]
        pages = int(state["pages_examined"])
        examined = int(state["files_examined"])
        downloaded = list(state["files_downloaded"])
    else:
        token = None
        pages = 0
        examined = 0
        downloaded: list[dict[str, object]] = []
    with api.build_kaggle_client() as client:
        while cli.expected_count is None or len(downloaded) < cli.expected_count:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            api._set_paging(request, cli.page_size, token)
            for retry in range(8):
                try:
                    response = client.kernels.kernels_api_client.list_kernel_session_output(request)
                    break
                except HTTPError as error:
                    if error.response.status_code != 429 or retry == 7:
                        raise
                    time.sleep(min(30.0, 2.0**retry))
            pages += 1
            for item in response.files or []:
                examined += 1
                if not item.file_name.startswith(cli.prefix):
                    continue
                pure = PurePosixPath(item.file_name)
                if pure.is_absolute() or ".." in pure.parts:
                    raise ValueError(f"unsafe output name: {item.file_name}")
                target = cli.output.joinpath(*pure.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                result = requests.get(item.url, timeout=120)
                result.raise_for_status()
                target.write_bytes(result.content)
                downloaded.append(
                    {
                        "path": item.file_name,
                        "bytes": target.stat().st_size,
                        "sha256": sha256(target),
                    }
                )
            token = response.next_page_token
            state = {
                "next_page_token": token,
                "pages_examined": pages,
                "files_examined": examined,
                "files_downloaded": downloaded,
            }
            state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            if not token:
                break
            time.sleep(cli.page_delay_seconds)
    if cli.expected_count is not None and len(downloaded) != cli.expected_count:
        raise RuntimeError(
            f"expected {cli.expected_count} files under {cli.prefix}, downloaded {len(downloaded)}"
        )
    manifest = {
        "schema_version": 1,
        "kernel_slug": slug,
        "prefix": cli.prefix,
        "pages_examined": pages,
        "files_examined": examined,
        "files_downloaded": downloaded,
    }
    manifest_path = cli.output / "subset_download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "kernel_slug": slug,
                "pages_examined": pages,
                "files_examined": examined,
                "files_downloaded": len(downloaded),
                "manifest_sha256": sha256(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
