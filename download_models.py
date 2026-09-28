"""Download the ONNX checkpoints listed in models/requirements.txt."""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
REQUIREMENTS = MODELS / "requirements.txt"


def _entries() -> list[tuple[str, str]]:
    entries = []
    for line in REQUIREMENTS.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, url = line.split()
        entries.append((name, url))
    return entries


def download(name: str, url: str) -> None:
    dest = MODELS / name
    if dest.exists():
        print(f"exists {dest.name}")
        return
    print(f"downloading {dest.name}")
    request = Request(url, headers={"User-Agent": "solid_lens"})
    with urlopen(request) as response, dest.open("wb") as out:
        while chunk := response.read(1 << 20):
            out.write(chunk)
    print(f"wrote {dest.name} ({dest.stat().st_size} bytes)")


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    for name, url in _entries():
        download(name, url)


if __name__ == "__main__":
    main()
