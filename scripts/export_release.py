"""Create a portable code archive, or a separate full local handoff archive."""
import argparse
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--with-assets", action="store_true", help="Include base weights, tokenizer and manifests (~6.8 GiB)")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    ignored = {".git", ".venv", ".pytest_cache", "__pycache__", "runs", "data_cache", "dist", "build",
               ".checkpoint_download"}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or path.resolve() == a.output.resolve():
            continue
        if any(part in ignored or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.suffix in (".pyc", ".gz", ".tar"):
            continue
        if relative.parts[0] in ("checkpoints", "assets") and not a.with_assets and path.name != "README.md":
            continue
        files.append(path)
    # No gzip for the large asset archive: avoids spending CPU recompressing model weights.
    with tarfile.open(a.output, "w" if a.with_assets else "w:gz") as archive:
        for path in sorted(files):
            archive.add(path, arcname=Path("openpi_navsim_sft") / path.relative_to(root), recursive=False)
    print(f"Created {a.output} ({a.output.stat().st_size} bytes, {len(files)} files)")


if __name__ == "__main__":
    main()
