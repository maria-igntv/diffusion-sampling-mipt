"""Build a small uploadable archive without datasets, weights or user documents."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "dist" / "diffusion-sampling-colab.zip"
    output.parent.mkdir(exist_ok=True)
    candidates = [root / "README.md", root / "pyproject.toml"]
    for directory in ("src", "configs", "docs", "notebooks", "tests", "scripts"):
        candidates.extend((root / directory).rglob("*"))
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(set(candidates)):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                archive.write(path, Path("diffusion-sampling") / path.relative_to(root))
    print(output)
    print(f"{output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
