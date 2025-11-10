#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

def build_template_miner() -> TemplateMiner:
    """
    Create a TemplateMiner with a self-contained default config.
    Works across Drain3 versions (no load_default() needed).
    """
    cfg = TemplateMinerConfig()
    # Try to load the library's DEFAULT_CONFIG if present, else use our own
    try:
        from drain3.template_miner_config import DEFAULT_CONFIG  # type: ignore
        cfg.load_from_str(DEFAULT_CONFIG)
    except Exception:
        cfg.load_from_str(
            # Minimal, safe defaults; tune sim_th/depth/max_children as needed
            """
[Masking]
mask_prefix=__mask_
masking=NullMasker

[Clusters]
max_clusters=200000

[Drain]
sim_th=0.5
depth=5
max_children=100

[Persistence]
enabled=False

[Profiling]
enabled=False
"""
        )
    # If you want to tweak parameters, do it by reloading with a modified INI string
    # or add setters here if your Drain3 version exposes them.
    return TemplateMiner(cfg)

def main():
    ap = argparse.ArgumentParser(description="Mine Drain3 templates from examples.txt")
    ap.add_argument("--input", default="examples.txt", help="Path to raw logs file")
    ap.add_argument("--templates_out", default="drain3_templates.jsonl",
                    help="Where to write one-JSON-per-template")
    ap.add_argument("--mapping_out", default="logs_to_templates.csv",
                    help="CSV mapping of log line -> template_id")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    tm = build_template_miner()
    templates = {}  # template_id -> dict(template, count)

    mapping_lines = []
    with in_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for idx, raw in enumerate(fin):
            line = raw.strip()
            if not line:
                continue
            r = tm.add_log_message(line)
            # r: {'cluster_id': '...', 'template_mined': '...', ...}
            tid = r["cluster_id"]
            tpl = tm.get_template(r["cluster_id"])
            if tid not in templates:
                templates[tid] = {"template_id": tid, "template": tpl, "count": 0}
            templates[tid]["count"] += 1
            mapping_lines.append((idx, tid, line))

    # write templates (jsonl)
    with Path(args.templates_out).open("w", encoding="utf-8") as fout:
        for tid, rec in templates.items():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write mapping (csv): line_idx,template_id,raw
    with Path(args.mapping_out).open("w", encoding="utf-8") as fout:
        fout.write("line_idx,template_id,raw\n")
        for i, tid, raw in mapping_lines:
            # escape quotes in raw
            safe = '"' + raw.replace('"', '""') + '"'
            fout.write(f"{i},{tid},{safe}\n")

    print(f"[OK] {len(templates)} templates → {args.templates_out}")
    print(f"[OK] {len(mapping_lines)} mappings → {args.mapping_out}")

if __name__ == "__main__":
    main()
