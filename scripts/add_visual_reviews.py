"""Add the inspected visual reviews to full exports and the comparison report."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ARCHIVES = {
    "ddim": "ddim_uniform_t_nfe39_full_final_fid_000000-005000_8ce6f444.zip",
    "dpmpp_2m": "dpmpp_2m_uniform_t_nfe39_full_final_fid_000000-005000_3376f091.zip",
    "heun": "heun_uniform_t_nfe39_full_final_fid_000000-005000_a018e253.zip",
}


def add_files(source, output, additions):
    """Publish a new verified ZIP with hashes covering all its data entries."""
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate output archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    with ZipFile(source) as original:
        names = original.namelist()
        if original.testzip() is not None or len(names) != len(set(names)):
            raise ValueError("Input archive is damaged or has duplicate entries")
        manifest_name = "export_manifest.json" if "export_manifest.json" in names else "review_manifest.json"
        metadata = json.loads(original.read(manifest_name)) if manifest_name in names else {}
        for name, expected in metadata.get("files_sha256", {}).items():
            if hashlib.sha256(original.read(name)).hexdigest() != expected:
                raise ValueError(f"Input checksum mismatch: {name}")
        hashes = {}
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=3) as archive:
            for name in names:
                if name == manifest_name or name in additions:
                    continue
                data = original.read(name)
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(name, data)
            for name, data in additions.items():
                hashes[name] = hashlib.sha256(data).hexdigest()
                archive.writestr(name, data)
            metadata["files_sha256"] = hashes
            archive.writestr(manifest_name, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Output CRC verification failed")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Output checksum mismatch: {name}")
    temporary.replace(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/reviewed_archives"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report_additions = {}
    for method, filename in ARCHIVES.items():
        note = root / "docs/visual_reviews" / f"{method}_uniform_t_nfe39.md"
        output = args.output_dir / filename.replace(".zip", "_reviewed.zip")
        add_files(args.input_dir / filename, output, {"run/visual_review.md": note.read_bytes()})
        report_additions[f"report/visual_reviews/{note.name}"] = note.read_bytes()
        print(output.resolve(), flush=True)
    for path in sorted((root / "docs/figures").glob("visual_review_nfe39_*.png")):
        report_additions[f"report/figures/{path.name}"] = path.read_bytes()
    report_additions["report/visual_review_summary.md"] = (root / "docs/study_uniform39_results.md").read_bytes()
    report = "study_uniform39_v1_report_79a5ad9c.zip"
    output = args.output_dir / report.replace(".zip", "_reviewed.zip")
    add_files(args.input_dir / report, output, report_additions)
    print(output.resolve(), flush=True)


if __name__ == "__main__":
    main()
