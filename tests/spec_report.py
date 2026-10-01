"""Print the spec's week table: every component, edge, flow and invariant in scope, with the test that
covers it and PASS / FAIL (rule 5 of the architecture spec).

    uv run python tests/spec_report.py --week 4        # runs the suite, prints the week-4 table
    uv run python tests/spec_report.py --week all
    uv run python tests/spec_report.py --junit out.xml # reuse an earlier pytest --junitxml run

A row is PASS when at least one test named for its ID exists and every such test passed. Hardware rows
that no software test can prove print MANUAL and point at the bench checklist in firmware/README.md.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WHAT = {
    "HW1": "clip mic", "HW2": "mute switch", "HW3": "listening LED", "HW4": "laptop mic", "HW5": "laptop signals",
    "HW6": "local data (.ics)", "HW7": "clip status LED", "A1": "mic capture", "A2": "ring buffer", "A3": "energy gate",
    "A4": "voice activity", "A5": "wake word", "A6": "clip serial link", "C1": "context fusion", "P1": "speech to text",
    "P2": "speaker tag", "P3": "sensitive filter", "P4": "segmenter", "P5": "signal extractor", "M1": "embedder",
    "M2": "encrypted memory", "M3": "consolidator", "R1": "intent router", "R2": "context builder", "R3": "local LLM",
    "R4": "tool executor", "R5": "confidence gate", "R6": "proactive scheduler", "R7": "conversation buffer",
    "O1": "status controller", "O2": "egress ledger view", "O3": "voice reply", "O4": "companion UI", "G1": "fact gateway",
    "X1": "Open-Meteo", "X2": "search API", "X3": "news feeds", "supervisor": "process supervisor",
    "guard": "network guard", "meta.history": "intent meta.history", "general.knowledge": "intent general.knowledge",
    "M2.turns": "turns table",
}  # fmt: skip
ALIASES = {  # IDs whose tests are named by a word rather than the ID
    "supervisor": "CL8", "guard": "INV3", "meta.history": "meta_history", "general.knowledge": "general_knowledge",
    "M2.turns": "E39",
}  # fmt: skip
MANUAL = {"HW1", "HW2", "HW3", "HW7", "E01", "E02", "E03"}  # need the physical clip: firmware/README.md checklist


def edges(*n: int) -> list[str]:
    return [f"E{i:02d}" for i in n]


WEEKS = {
    1: ["HW4", "A1", "A2", "A3", "A4", "A5", "P1", "O1", "O4", "supervisor", "guard", *edges(4, 6, 7, 8, 9, 10, 36, 37, 41),
        "INV1", "INV2", "INV3", "INV4", "INV5", "INV9", "INV10"],
    2: ["HW5", "C1", "P2", "P3", "P4", "P5", "M1", "M2", "M3", "O4", *edges(11, 12, 13, 14, 15, 16, 17, 18, 32, 38),
        "F1", "F8", "F7", "INV6"],
    3: ["HW6", "C1", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "G1", "O4", "M3", "M2.turns", "meta.history",
        "general.knowledge", *edges(19, 20, 21, 22, 23, 24, 28, 30, 31, 33, 34, 35, 39, 40), "F2", "F6", "F7", "F9"],
    4: ["HW1", "HW2", "HW3", "HW7", "A6", "G1", "O2", "O3", "X1", "X2", "X3", *edges(1, 2, 3, 5, 25, 26, 27, 29, 37),
        "F3", "F4", "F5", "INV3", "INV7", "INV8"],
}  # fmt: skip
DEGRADE = ["no_network", "nothing_in_memory", "conflicting_memories", "beyond_the_model", "low_memory", "battery_low",
           "clip_unplugged"]  # sheet 8 rows; the crashed-process row is CL8


def run_pytest() -> dict[str, str]:
    junit = ROOT / "runs" / "spec-report.xml"
    junit.parent.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", f"--junitxml={junit}",
         "--ignore=tests/test_live_mic.py"],
        cwd=ROOT,
    )
    return read_junit(junit)


def read_junit(path: Path) -> dict[str, str]:
    out = {}
    for case in ET.parse(path).getroot().iter("testcase"):
        name = case.get("name", "")
        if case.find("failure") is not None or case.find("error") is not None:
            state = "FAIL"
        elif case.find("skipped") is not None:
            state = "SKIP"
        else:
            state = "PASS"
        out[name] = state
    return out


def tests_for(item: str, results: dict[str, str]) -> list[tuple[str, str]]:
    pat = re.escape(ALIASES.get(item, item))
    rx = re.compile(rf"^test_(?:.*_)?{pat}(?:_|\[|$)")
    return [(n, s) for n, s in results.items() if rx.search(n)]


def verdict(item: str, found: list[tuple[str, str]]) -> str:
    if item in MANUAL and not found:
        return "MANUAL"
    if not found:
        return "NO TEST"
    states = {s for _, s in found}
    if "FAIL" in states:
        return "FAIL"
    return "PASS" if "PASS" in states else "SKIP"


def report(week: int, results: dict[str, str]) -> int:
    print(f"\n=== Week {week} ===")
    print(f"{'ID':<18}{'what':<26}{'test':<80}result")
    bad = 0
    for item in WEEKS[week]:
        found = tests_for(item, results)
        v = verdict(item, found)
        name = found[0][0] if found else ("(bench checklist, firmware/README.md)" if v == "MANUAL" else "-")
        more = f" (+{len(found) - 1})" if len(found) > 1 else ""
        print(f"{item:<18}{WHAT.get(item, ''):<26}{(name + more)[:78]:<80}{v}")
        bad += v not in ("PASS", "MANUAL")
    if week == 4:
        for d in DEGRADE:
            found = [(n, s) for n, s in results.items() if n.startswith(f"test_degrade_{d}")]
            v = verdict(d, found)
            print(f"{'degrade':<18}{d:<26}{(found[0][0] if found else '-')[:78]:<80}{v}")
            bad += v != "PASS"
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="all")
    ap.add_argument("--junit", help="use an existing junit xml instead of running pytest")
    args = ap.parse_args()
    results = read_junit(Path(args.junit)) if args.junit else run_pytest()
    weeks = list(WEEKS) if args.week == "all" else [int(args.week)]
    bad = sum(report(w, results) for w in weeks)
    print(f"\n{'ALL ROWS PASS (MANUAL rows need the clip)' if not bad else f'{bad} row(s) not passing'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
