"""Command-line interface: ``python -m fewshotlab <command>``.

The CLI only parses arguments and calls library functions, so the Streamlit app and the CLI
share one implementation. Commands that a later phase implements exit with code 3 and write
nothing, so an unfinished command can never look like a success.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from fewshotlab import PROTOCOL_ID, __version__
from fewshotlab.errors import FewShotLabError, NotImplementedPhaseError
from fewshotlab.status import ExitCode

DATASET_CHOICES = ("flowers102", "neu_cls")
DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")
SPLIT_CHOICES = ("val", "test")

EPILOG = """\
exit codes:
  0  success
  1  the command ran and failed (validation error, bad input)
  2  usage error
  3  not implemented yet (planned for a later phase; nothing was written)
  4  blocked by a missing prerequisite (dataset source, weights, device)
  5  refused by a protocol guard (for example, test access before the plan is sealed)
"""

Handler = Callable[[argparse.Namespace], int]


def _not_implemented(command: str, phase: str, purpose: str) -> Handler:
    def handler(_args: argparse.Namespace) -> int:
        raise NotImplementedPhaseError(command, phase, purpose)

    return handler


def _run_doctor(args: argparse.Namespace) -> int:
    from fewshotlab.doctor import run_doctor

    return run_doctor(as_json=args.json, require_device=args.require_device)


def _run_data_prepare(args: argparse.Namespace) -> int:
    from fewshotlab.data.prepare import prepare_dataset

    record = prepare_dataset(
        args.dataset,
        source=args.source,
        allow_download=args.allow_download,
        mirror=args.mirror,
        cross_check=args.cross_check,
    )
    print(f"{args.dataset}: {record['status']}: {record['summary']}")
    print(f"manifest: {record['manifest']['path']} sha256 {record['manifest']['sha256']}")
    print(f"next: python -m fewshotlab data audit --dataset {args.dataset}")
    return ExitCode.OK


def _run_data_audit(args: argparse.Namespace) -> int:
    from fewshotlab.data.audit import run_audit
    from fewshotlab.data.registry import audit_dir
    from fewshotlab.paths import get_paths

    report = run_audit(args.dataset, near_dup_threshold=args.near_dup_threshold)
    near = report["near_duplicates"]
    exact = report["exact_duplicates"]
    print(
        f"{args.dataset} audit: {report['status']} ({report['manifest']['rows']} images, "
        f"manifest sha256 {report['manifest']['sha256'][:12]})"
    )
    print(
        f"  integrity failures {len(report['integrity_failures'])}, decode failures "
        f"{len(report['decode_failures'])}"
    )
    print(
        f"  exact-duplicate groups {exact['pixel_identical_groups']} "
        f"(cross-split {len(exact['cross_split_groups'])})"
    )
    print(
        f"  near-duplicate candidates <= {near['threshold']} bits: "
        f"{near['pairs_within_threshold']} (cross-split {near['cross_split_pairs']})"
    )
    for finding in report["findings"]:
        print(f"  finding: {finding}")
    print(f"  report: {audit_dir(get_paths(), args.dataset) / 'data_audit.md'}")
    return ExitCode.OK if report["status"] == "PASS" else ExitCode.FAILURE


def _run_support_build(args: argparse.Namespace) -> int:
    from fewshotlab.data.support import build_from_config

    summary = build_from_config(args.config, dataset=args.dataset)
    for name, info in summary["datasets"].items():
        if info["status"] != "BUILT":
            print(f"{name}: {info['status']}; no supports built (prepare and audit it first)")
            continue
        states: dict[str, int] = {}
        for item in info["built"]:
            states[item["state"]] = states.get(item["state"], 0) + 1
        print(
            f"{name}: {len(info['built'])} support manifests for K={list(summary['shots'])} "
            f"x seeds={list(summary['seeds'])} ({states})"
        )
    print(f"label budget: {summary['label_budget_report']}")
    if summary["missing"]:
        print(f"blocked datasets: {summary['missing']}", file=sys.stderr)
        return ExitCode.BLOCKED
    return ExitCode.OK


def _run_weights_fetch(args: argparse.Namespace) -> int:
    from fewshotlab.models.backbone import ensure_imagenet_weights
    from fewshotlab.paths import get_paths

    path, info = ensure_imagenet_weights(get_paths().weights_dir, allow_download=True)
    print(f"ResNet18 {info.weights_id}: {path}")
    print(f"sha256 {info.sha256} (matches torchvision's file-name prefix)")
    return ExitCode.OK


def _command(
    sub: argparse._SubParsersAction, name: str, label: str, help_text: str, handler: Handler
) -> argparse.ArgumentParser:
    parser = sub.add_parser(
        name,
        help=help_text,
        description=help_text,
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(handler=handler, command_label=label)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fewshotlab",
        description=(
            "FewShotLab: reproducible transfer-learning experiments with few labeled images "
            f"(protocol {PROTOCOL_ID})."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"fewshotlab {__version__} (protocol {PROTOCOL_ID})"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    p = _command(
        sub,
        "doctor",
        "doctor",
        "Report environment, devices, caches, data and ledger status (read-only).",
        _run_doctor,
    )
    p.add_argument("--json", action="store_true", help="print the full report as JSON")
    p.add_argument(
        "--require-device",
        choices=("cpu", "mps", "cuda"),
        default=None,
        help="exit with code 4 and a fix-it message if this device is not usable",
    )

    data = sub.add_parser(
        "data",
        help="Dataset preparation and audits.",
        description="Dataset preparation and audits.",
    )
    data_sub = data.add_subparsers(dest="data_command", metavar="<data-command>", required=True)
    p = _command(
        data_sub,
        "prepare",
        "data prepare",
        "Acquire an original dataset and write its immutable manifest. Exits 4 and "
        "records BLOCKED when the original source is unreachable.",
        _run_data_prepare,
    )
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument(
        "--source",
        metavar="PATH",
        default=None,
        help="use a user-supplied original archive or folder instead of downloading",
    )
    p.add_argument(
        "--allow-download",
        action="store_true",
        help="permit fetching the original files from the official source",
    )
    p.add_argument(
        "--mirror",
        choices=("internet-archive",),
        default=None,
        help="flowers102 only: fetch Internet Archive captures of the official URLs (needs "
        "--allow-download; files are kept only if their MD5 matches torchvision's record)",
    )
    p.add_argument(
        "--cross-check",
        metavar="PATH",
        default=None,
        help="flowers102 only: compare a re-packaged copy (e.g. Kaggle) with the official files; "
        "the copy is only read, never used to build the manifest",
    )
    p = _command(
        data_sub,
        "audit",
        "data audit",
        "Audit integrity, exact duplicates, near-duplicate candidates and split "
        "separation; writes reports/data_audit/<dataset>/.",
        _run_data_audit,
    )
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    p.add_argument(
        "--near-dup-threshold",
        type=int,
        default=4,
        metavar="BITS",
        help="maximum pHash Hamming distance (of 64 bits) listed as a near-duplicate "
        "candidate for manual review (default: 4)",
    )

    support = sub.add_parser(
        "support",
        help="K-shot support-set construction.",
        description="K-shot support-set construction.",
    )
    support_sub = support.add_subparsers(
        dest="support_command", metavar="<support-command>", required=True
    )
    p = _command(
        support_sub,
        "build",
        "support build",
        "Build nested K-shot support manifests for every seed (requires a prepared and "
        "audited dataset).",
        _run_support_build,
    )
    p.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="protocol config listing datasets, shots and seeds (e.g. configs/protocol-v1.yaml)",
    )
    p.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        default=None,
        help="limit to one dataset (default: every dataset in the config)",
    )

    weights = sub.add_parser(
        "weights",
        help="Pretrained weight cache.",
        description="Pretrained weight cache.",
    )
    weights_sub = weights.add_subparsers(
        dest="weights_command", metavar="<weights-command>", required=True
    )
    _command(
        weights_sub,
        "fetch",
        "weights fetch",
        "Download ResNet18 IMAGENET1K_V1 weights into the project cache once (an explicit "
        "download; verified against torchvision's SHA-256 file-name prefix).",
        _run_weights_fetch,
    )

    p = _command(
        sub,
        "train",
        "train",
        "Fit one run from a config (support + validation only; never test).",
        _not_implemented("train", "P3", "training engine"),
    )
    p.add_argument("--config", required=True, metavar="PATH", help="run config (YAML)")
    p.add_argument(
        "--device", choices=DEVICE_CHOICES, default=None, help="override the config's device"
    )
    p.add_argument(
        "--allow-download", action="store_true", help="permit fetching missing pretrained weights"
    )

    p = _command(
        sub,
        "evaluate",
        "evaluate",
        "Evaluate a stored run on validation, or on test only through the sealed gate.",
        _not_implemented("evaluate", "P3", "evaluation"),
    )
    p.add_argument("--run", required=True, metavar="RUN_ID")
    p.add_argument("--split", required=True, choices=SPLIT_CHOICES)
    p.add_argument("--device", choices=DEVICE_CHOICES, default=None)

    p = _command(
        sub,
        "benchmark",
        "benchmark",
        "Plan or execute a prespecified experiment matrix.",
        _not_implemented("benchmark", "P4", "experiment matrix execution"),
    )
    p.add_argument("--config", required=True, metavar="PATH", help="study config (YAML)")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="list planned slots and duplicates without running anything",
    )

    p = _command(
        sub,
        "lock-test",
        "lock-test",
        "Seal the evaluation plan and record an audit token before test access.",
        _not_implemented("lock-test", "P5", "test-gate sealing"),
    )
    p.add_argument("--plan", required=True, metavar="PATH", help="full-study plan (YAML)")

    p = _command(
        sub,
        "report",
        "report",
        "Generate measured tables and plots from the ledger and stored predictions.",
        _not_implemented("report", "P4", "measured reports"),
    )
    p.add_argument("--ledger", default="runs/ledger.jsonl", metavar="PATH")
    p.add_argument("--out", default="reports", metavar="DIR")

    p = _command(
        sub,
        "demo",
        "demo",
        "Launch the offline Streamlit demonstration.",
        _not_implemented("demo", "P6", "Streamlit demonstration"),
    )
    p.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except FewShotLabError as exc:
        print(f"fewshotlab {args.command_label}: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
