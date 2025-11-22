#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence


def build_template_miner() -> TemplateMiner:
    """
    Create a TemplateMiner using the current Drain3 API.

    - Uses TemplateMinerConfig defaults.
    - If a local drain3.ini exists next to this script and the config
      supports .load(), it will be loaded.
    - Uses FilePersistence to persist miner state.
    """
    cfg = TemplateMinerConfig()

    # Try to load a local drain3.ini (optional)
    ini_path = Path(__file__).resolve().parent / "drain3.ini"
    if hasattr(cfg, "load") and ini_path.exists():
        print(f"[Drain3] Loading config from {ini_path}")
        cfg.load(str(ini_path))
    else:
        if not ini_path.exists():
            print(f"[Drain3] No drain3.ini found at {ini_path}, using defaults")
        else:
            print("[Drain3] TemplateMinerConfig has no .load(); using defaults")

    # Persist templates so re-runs can reuse clusters
    persistence = FilePersistence("drain3_state.bin")

    # Newer Drain3: TemplateMiner(persistence, config=cfg)
    tm = TemplateMiner(persistence, config=cfg)
    return tm


def main():
    ap = argparse.ArgumentParser(description="Mine Drain3 templates from examples.txt")
    ap.add_argument("--input", default="examples.txt", help="Path to raw logs file")
    ap.add_argument(
        "--templates_out",
        default="drain3_templates.jsonl",
        help=(
            "Where to write one-JSON-per-template "
            "(default is domain-suffixed, e.g. drain3_templates_openstack.jsonl)"
        ),
    )
    ap.add_argument(
        "--mapping_out",
        default="logs_to_templates.csv",
        help=(
            "CSV mapping of log line -> template_id "
            "(default is domain-suffixed, e.g. logs_to_templates_openstack.csv)"
        ),
    )
    ap.add_argument(
        "--domain_id",
        default="generic",
        help="Logical domain for these logs (e.g. generic, openstack, spark)",
    )
    args = ap.parse_args()

    domain = (args.domain_id or "generic").lower()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # If user left defaults, automatically suffix with domain
    # so you get logs_to_templates_openstack.csv, logs_to_templates_spark.csv, etc.
    default_templates = "drain3_templates.jsonl"
    default_mapping = "logs_to_templates.csv"

    if args.templates_out == default_templates:
        templates_out_path = Path(f"drain3_templates_{domain}.jsonl")
    else:
        templates_out_path = Path(args.templates_out)

    if args.mapping_out == default_mapping:
        mapping_out_path = Path(f"logs_to_templates_{domain}.csv")
    else:
        mapping_out_path = Path(args.mapping_out)

    print(
        f"[domain] domain_id='{domain}' "
        f"→ templates_out={templates_out_path.name}, mapping_out={mapping_out_path.name}"
    )

    tm = build_template_miner()
    templates = {}  # template_id -> dict(template, count)

    # mapping_lines now stores: (line_idx, template_id, template_text, raw)
    mapping_lines = []

    with in_path.open("r", encoding="utf-8", errors="ignore") as fin:
        for idx, raw in enumerate(fin):
            line = raw.strip()
            if not line:
                continue

            r = tm.add_log_message(line)
            # r: {'cluster_id': '...', 'template_mined': '...', ...}
            tid = r["cluster_id"]
            tpl = r["template_mined"]

            if tid not in templates:
                templates[tid] = {"template_id": tid, "template": tpl, "count": 0}
            templates[tid]["count"] += 1

            # Store template text along with mapping
            mapping_lines.append((idx, tid, tpl, line))

    # write templates (jsonl)
    with templates_out_path.open("w", encoding="utf-8") as fout:
        for tid, rec in templates.items():
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # write mapping (csv): line_idx,template_id,template_text,raw,domain_id
    with mapping_out_path.open("w", encoding="utf-8") as fout:
        fout.write("line_idx,template_id,template_text,raw,domain_id\n")
        for i, tid, tpl, raw in mapping_lines:
            # escape quotes in raw and template
            safe_raw = '"' + raw.replace('"', '""') + '"'
            safe_tpl = '"' + tpl.replace('"', '""') + '"'
            fout.write(f"{i},{tid},{safe_tpl},{safe_raw},{domain}\n")

    print(f"[OK] {len(templates)} templates → {templates_out_path}")
    print(f"[OK] {len(mapping_lines)} mappings → {mapping_out_path}")


if __name__ == "__main__":
    main()
