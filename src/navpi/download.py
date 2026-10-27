"""Public official GCS checkpoint download, verified before atomic publication."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode, quote
from urllib.request import urlopen

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_base/params/"


def verify_object(path, metadata):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != int(metadata["size"]):
        return False
    if metadata.get("md5Hash"):
        digest = hashlib.md5()
        expected = metadata["md5Hash"]
    else:
        import google_crc32c
        digest = google_crc32c.Checksum()
        expected = metadata["crc32c"]
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode() == expected


def download_official(destination, workers=4):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    entries, page = [], None
    while True:
        query = {"prefix": PREFIX, "maxResults": 1000}
        if page:
            query["pageToken"] = page
        url = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?{urlencode(query)}"
        with urlopen(url, timeout=120) as response:
            listing = json.load(response)
        entries.extend(row for row in listing.get("items", []) if not row["name"].endswith("/"))
        page = listing.get("nextPageToken")
        if not page:
            break
    if not entries:
        raise RuntimeError("Official checkpoint listing is empty")

    def fetch(metadata):
        relative = Path(metadata["name"].removeprefix("checkpoints/pi05_base/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid object path")
        target = destination / relative
        if verify_object(target, metadata):
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://storage.googleapis.com/{BUCKET}/{quote(metadata['name'], safe='/')}"
        url += "?" + urlencode({"generation": metadata["generation"]})
        partial = target.with_name(target.name + ".download")
        for attempt in range(3):
            try:
                with urlopen(url, timeout=120) as response, partial.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=8 << 20)
                if not verify_object(partial, metadata):
                    raise ValueError(f"Download checksum mismatch: {relative}")
                partial.replace(target)
                print(f"Downloaded {relative}", flush=True)
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, entries))
    (destination / "download_manifest.json").write_text(json.dumps(entries, indent=2))
    return destination / "params"
