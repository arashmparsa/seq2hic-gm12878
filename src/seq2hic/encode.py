"""ENCODE portal discovery and download.

File URLs are discovered through the REST API rather than hardcoded, because
ENCODE re-releases files and revokes old ones -- a pinned URL rots quietly and
you find out three hours into a download. `list` shows you what exists today;
`fetch` pulls it with resume + md5 verification.

CLI
---
    python -m seq2hic.encode list ENCSR830NVY
    python -m seq2hic.encode fetch ENCSR637XSC --output-type "fold change over control"
    python -m seq2hic.encode genome
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

PORTAL = "https://www.encodeproject.org"

ACCESSIONS = {
    "hic": "ENCSR830NVY",       # Hi-C, GM12878, Aiden lab (series ENCSR392PEL)
    "atac": "ENCSR637XSC",      # ATAC-seq, GM12878, Snyder lab, ENCODE4
}

# Auxiliary references. GENCODE/UCSC rather than the portal, since the portal's
# reference files are keyed by pipeline version and awkward to pin.
GENOME_FASTA_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
    "GRCh38.primary_assembly.genome.fa.gz"
)
GENCODE_GTF_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/"
    "gencode.v44.basic.annotation.gtf.gz"
)


def _get_json(url: str) -> dict:
    resp = requests.get(url, headers={"accept": "application/json"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def experiment_files(accession: str) -> list[dict]:
    """Return released files for an experiment, newest-status first."""
    meta = _get_json(f"{PORTAL}/experiments/{accession}/?format=json")
    files = []
    for f in meta.get("files", []):
        if f.get("status") not in ("released", "in progress"):
            continue
        files.append(
            {
                "accession": f.get("accession"),
                "output_type": f.get("output_type"),
                "file_format": f.get("file_format"),
                "file_format_type": f.get("file_format_type"),
                "assembly": f.get("assembly"),
                "biological_replicates": f.get("biological_replicates"),
                "file_size": f.get("file_size"),
                "md5sum": f.get("md5sum"),
                "href": f.get("href"),
                "status": f.get("status"),
            }
        )
    files.sort(key=lambda d: (d["output_type"] or "", d["file_format"] or ""))
    return files


def cmd_list(args: argparse.Namespace) -> int:
    acc = ACCESSIONS.get(args.accession, args.accession)
    files = experiment_files(acc)
    if not files:
        print(f"no released files found for {acc}", file=sys.stderr)
        return 1

    print(f"\n{acc}  ({len(files)} files)\n")
    hdr = f"{'file':<14}{'format':<12}{'assembly':<10}{'size':>10}  output_type"
    print(hdr)
    print("-" * (len(hdr) + 20))
    for f in files:
        size = f["file_size"] or 0
        size_s = f"{size / 1e9:.2f} GB" if size > 1e9 else f"{size / 1e6:.0f} MB"
        print(
            f"{f['accession'] or '?':<14}{f['file_format'] or '?':<12}"
            f"{f['assembly'] or '-':<10}{size_s:>10}  {f['output_type']}"
        )
    print()
    return 0


def _md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_md5: str | None = None) -> Path:
    """Resumable download with an md5 check. Skips if already complete and valid."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expected_md5:
        if _md5(dest) == expected_md5:
            print(f"  [have] {dest.name}")
            return dest
        print(f"  [bad md5, re-downloading] {dest.name}")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}

    with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
        if resp.status_code == 416:  # already fully fetched
            have_total = have
        else:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length", 0)) + have
            mode = "ab" if have else "wb"
            with open(part, mode) as fh, tqdm(
                total=total, initial=have, unit="B", unit_scale=True,
                desc=f"  {dest.name}", leave=False,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
            have_total = part.stat().st_size

    part.replace(dest)
    if expected_md5:
        got = _md5(dest)
        if got != expected_md5:
            raise RuntimeError(
                f"md5 mismatch for {dest.name}: expected {expected_md5}, got {got}"
            )
    print(f"  [ok] {dest.name} ({have_total if have else dest.stat().st_size} bytes)")
    return dest


def cmd_fetch(args: argparse.Namespace) -> int:
    acc = ACCESSIONS.get(args.accession, args.accession)
    files = experiment_files(acc)

    sel = files
    if args.output_type:
        sel = [f for f in sel if f["output_type"] == args.output_type]
    if args.file_format:
        sel = [f for f in sel if f["file_format"] == args.file_format]
    if args.assembly:
        sel = [f for f in sel if f["assembly"] == args.assembly]

    if not sel:
        print(
            f"no files in {acc} matched your filters. Run "
            f"`python -m seq2hic.encode list {acc}` to see what exists.",
            file=sys.stderr,
        )
        return 1

    outdir = Path(args.out)
    print(f"{acc}: downloading {len(sel)} file(s) -> {outdir}")
    manifest_path = outdir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for f in sel:
        url = PORTAL + f["href"]
        dest = outdir / Path(f["href"]).name
        download(url, dest, f["md5sum"])
        manifest[f["accession"]] = {**f, "local_path": str(dest), "experiment": acc}

    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {manifest_path}")
    return 0


def cmd_genome(args: argparse.Namespace) -> int:
    outdir = Path(args.out)
    print("reference genome + annotation")
    fa_gz = download(GENOME_FASTA_URL, outdir / "GRCh38.primary_assembly.genome.fa.gz")
    gtf_gz = download(GENCODE_GTF_URL, outdir / "gencode.v44.basic.annotation.gtf.gz")

    fa = outdir / "GRCh38.primary_assembly.genome.fa"
    if not fa.exists():
        print("  decompressing FASTA (this takes a few minutes)...")
        import gzip
        import shutil

        with gzip.open(fa_gz, "rb") as src, open(fa, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 24)
        print(f"  [ok] {fa.name}")

    tss = outdir / "gencode.tss.bed"
    if not tss.exists():
        _gtf_to_tss(gtf_gz, tss)
    return 0


def _gtf_to_tss(gtf_gz: Path, out_bed: Path) -> None:
    """Extract one TSS per gene (BED6: chrom start end gene_name score strand)."""
    import gzip

    seen: set[str] = set()
    n = 0
    with gzip.open(gtf_gz, "rt", encoding="utf-8") as fh, open(
        out_bed, "w", encoding="utf-8"
    ) as out:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom, start, end, strand, attrs = parts[0], int(parts[3]) - 1, int(parts[4]), parts[6], parts[8]
            name = _attr(attrs, "gene_name") or _attr(attrs, "gene_id") or "."
            if name in seen:
                continue
            seen.add(name)
            tss = start if strand == "+" else end - 1
            out.write(f"{chrom}\t{tss}\t{tss + 1}\t{name}\t0\t{strand}\n")
            n += 1
    print(f"  [ok] {out_bed.name} ({n} genes)")


def _attr(attrs: str, key: str) -> str | None:
    token = f'{key} "'
    i = attrs.find(token)
    if i < 0:
        return None
    j = attrs.find('"', i + len(token))
    return attrs[i + len(token) : j]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="seq2hic.encode", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list released files for an ENCODE experiment")
    pl.add_argument("accession", help="ENCSR... or the alias 'hic' / 'atac'")
    pl.set_defaults(func=cmd_list)

    pf = sub.add_parser("fetch", help="download matching files")
    pf.add_argument("accession")
    pf.add_argument("--output-type", default=None)
    pf.add_argument("--file-format", default=None)
    pf.add_argument("--assembly", default="GRCh38")
    pf.add_argument("--out", default="data/raw")
    pf.set_defaults(func=cmd_fetch)

    pg = sub.add_parser("genome", help="download GRCh38 FASTA + GENCODE TSS")
    pg.add_argument("--out", default="data/raw")
    pg.set_defaults(func=cmd_genome)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
