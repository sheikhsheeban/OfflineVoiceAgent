"""`kin` command line.

kin fetch-models            download models once, sha256-checked
kin run [--profile phone]   start listening, UI on 127.0.0.1
kin replay demo.wav         feed a recording through the same pipeline
kin bench demo.wav          headless replay; prints latency, CPU and RAM
kin search "budget"         search memory from the terminal
kin ask "when is the site visit?"   ask Kin from the terminal (router, tools, model, gate)
kin asr-test                say a sentence; every installed speech model transcribes it
kin devices                 list microphones
kin doctor                  check models, hashes and audio input
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import statistics
import sys
import time
import webbrowser
from pathlib import Path

from kin.config import MODELS_DIR, RUNS_DIR, AudioSettings, Settings, get_profile, load_model_spec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _entries(asr: list[str] | None = None) -> list[tuple[str, dict]]:
    """Model entries to check or fetch: everything fixed, plus the named ASR models
    (default: the phone profile's)."""
    spec = load_model_spec()
    names = asr if asr is not None else [spec["asr"]["phone"]]
    if names == ["all"]:
        names = list(spec["asr_models"])
    out = [(k, v) for k, v in spec.items() if isinstance(v, dict) and "files" in v]
    for n in names:
        if n not in spec["asr_models"]:
            raise SystemExit(f"unknown ASR model {n!r}; choose from {', '.join(spec['asr_models'])}")
        out.append((f"asr:{n}", spec["asr_models"][n]))
    return out


def installed_asr_models() -> list[str]:
    spec = load_model_spec()["asr_models"]
    return [n for n, m in spec.items() if all((MODELS_DIR / m["dir"] / f).exists() for f in m["files"])]


def verify_models(asr: list[str] | None = None) -> list[str]:
    """Return a list of problems; empty means every pinned file is present and intact."""
    problems = []
    for section, spec in _entries(asr):
        for name, digest in spec["files"].items():
            p = MODELS_DIR / spec["dir"] / name
            if not p.exists():
                problems.append(f"{section}: missing {p}")
            elif _sha256(p) != digest:
                problems.append(f"{section}: hash mismatch for {p}")
    return problems


def cmd_fetch_models(args) -> int:
    # The only place Kin downloads anything; the running app never does.
    import tarfile
    import tempfile
    import urllib.request

    for section, spec in _entries(args.asr):
        d = MODELS_DIR / spec["dir"]
        missing = [n for n, h in spec["files"].items() if not (d / n).exists() or _sha256(d / n) != h]
        if not missing:
            print(f"ok       {section}: {spec['name']}")
            continue
        print(f"fetching {section}: {spec['name']}")
        d.mkdir(parents=True, exist_ok=True)
        if "urls" in spec:
            for n in missing:
                urllib.request.urlretrieve(spec["urls"][n], d / n)
        elif "url_base" in spec:
            for n in missing:
                urllib.request.urlretrieve(spec["url_base"] + n, d / n)
        elif spec["url"].endswith((".tar.bz2", ".tar.gz")):
            dest = MODELS_DIR / spec["extract_to"]
            with tempfile.TemporaryDirectory() as tmp:
                arc = Path(tmp) / "model.tar"
                urllib.request.urlretrieve(spec["url"], arc)
                with tarfile.open(arc) as tf:
                    for m in tf.getmembers():  # refuse path traversal
                        target = (dest / m.name).resolve()
                        if not str(target).startswith(str(dest.resolve())) or m.issym() or m.islnk():
                            raise SystemExit(f"unsafe path in archive: {m.name}")
                    tf.extractall(dest)
            if spec.get("prune"):
                for f in d.iterdir():
                    if f.is_file() and f.name not in spec["files"]:
                        f.unlink()
        else:
            urllib.request.urlretrieve(spec["url"], d / missing[0])
    problems = verify_models(args.asr)
    for p in problems:
        print("ERROR", p)
    if not problems and args.asr is None:
        print("other speech models: uv run kin fetch-models --asr parakeet-0.6b   (or --asr all)")
    return 1 if problems else 0


def _settings(args) -> Settings:
    audio = AudioSettings(device=getattr(args, "device", None))
    return Settings(
        profile=get_profile(args.profile),
        audio=audio,
        ui_port=getattr(args, "port", 8765),
        memory_db=getattr(args, "db", None),
        asr_model=getattr(args, "asr", None),
        calendar_path=getattr(args, "calendar", None),
        warm_llm=getattr(args, "warm", False),
    )


def _require_models(args=None) -> None:
    from kin.config import asr_model_name

    asr = None
    if args is not None and hasattr(args, "profile"):
        asr = [asr_model_name(get_profile(args.profile).name, getattr(args, "asr", None))]
    problems = verify_models(asr)
    if problems:
        for p in problems:
            print("ERROR", p, file=sys.stderr)
        hint = f" --asr {asr[0]}" if asr else ""
        raise SystemExit(f"models missing or modified: run `uv run kin fetch-models{hint}`")


def _run(args, source: str, speed: float, until_eos: bool) -> int:
    from kin.logs import setup
    from kin.supervisor import Supervisor

    setup("supervisor")
    _require_models(args)
    csv_path = RUNS_DIR / f"metrics-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    sup = Supervisor(_settings(args), source=source, speed=speed, ui=True, metrics_csv=csv_path)
    try:
        sup.start()
        logging.getLogger("kin").info("open %s", sup.ui_url)
        if not args.no_browser:
            webbrowser.open(sup.ui_url)
        sup.run(until_eos=until_eos and not getattr(args, "stay", False))
    except KeyboardInterrupt:
        pass
    finally:
        sup.stop()
        _print_stats_summary(sup.stats_history)
        print(f"metrics written to {csv_path}")
    return 0


def cmd_run(args) -> int:
    return _run(args, "mic", 1.0, until_eos=False)


def cmd_replay(args) -> int:
    return _run(args, str(Path(args.wav).resolve()), args.speed, until_eos=True)


def _print_stats_summary(history: list[dict], skip_first: int = 2) -> dict:
    rows = history[skip_first:] or history
    if not rows:
        return {}
    summary = {}
    print("\nprocess      CPU avg %  CPU max %   RAM avg MB  RAM max MB   (CPU % per core)")
    for name in rows[0]["procs"]:
        cpu = [r["procs"][name]["cpu"] for r in rows if name in r["procs"]]
        ram = [r["procs"][name]["rss_mb"] for r in rows if name in r["procs"]]
        summary[name] = {
            "cpu_avg": statistics.mean(cpu),
            "cpu_max": max(cpu),
            "rss_avg": statistics.mean(ram),
            "rss_max": max(ram),
        }
        s = summary[name]
        print(f"{name:<12} {s['cpu_avg']:9.1f} {s['cpu_max']:10.1f} {s['rss_avg']:12.0f} {s['rss_max']:11.0f}")
    tot_cpu = [r["cpu_machine_pct"] for r in rows]
    tot_ram = [r["rss_mb_total"] for r in rows]
    summary["total"] = {"cpu_machine_avg": statistics.mean(tot_cpu), "rss_total_max": max(tot_ram)}
    print(f"{'all':<12} machine CPU avg {statistics.mean(tot_cpu):.2f}%   RAM total max {max(tot_ram):.0f} MB")
    return summary


def cmd_bench(args) -> int:
    from kin.logs import setup
    from kin.supervisor import Supervisor

    setup("supervisor")
    _require_models(args)
    wav = str(Path(args.wav).resolve()) if args.wav else "mic"
    sup = Supervisor(_settings(args), source=wav, speed=args.speed, ui=False)
    t0 = time.perf_counter()
    try:
        sup.start()
        t_ready = time.perf_counter() - t0
        sup.run(until_eos=wav != "mic", duration=args.seconds if wav == "mic" else None)
    finally:
        sup.stop()
    msgs = [m for m in sup.messages if m.get("type") in ("transcript", "wake")]
    tr = [m for m in msgs if m["type"] == "transcript"]
    print(f"\nstart-up to core ready: {t_ready:.2f} s   profile: {sup.settings.profile.name}")
    for m in msgs:
        if m["type"] == "wake":
            print(f"  WAKE  score={m['score']}")
        else:
            tag = "CMD " if m["wake"] else "    "
            when = f"[{m['start_s']:6.2f}s +{m['audio_s']:.2f}s]"
            print(f"  {tag} {when} asr={m['asr_ms']:.0f}ms rtf={m['rtf']:.3f}  {m['text']}")
    if tr:
        print(
            f"ASR: mean {statistics.mean(m['asr_ms'] for m in tr):.0f} ms, "
            f"mean RTF {statistics.mean(m['rtf'] for m in tr):.3f}, "
            f"end-of-speech->text max {max(m['latency_ms'] for m in tr):.0f} ms"
        )
    summary = _print_stats_summary(sup.stats_history)
    if args.json:
        Path(args.json).write_text(json.dumps({"messages": msgs, "resources": summary}, indent=2))
    return 0


def cmd_ask(args) -> int:
    from kin.config import default_calendar_path
    from kin.core.process import open_reasoner
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store

    _require_models()
    mem = Memory(Store.open(args.db), Embedder())
    try:
        reasoner = open_reasoner(mem, get_profile(args.profile), args.calendar or default_calendar_path())
        ans = reasoner.handle(args.question)
    finally:
        mem.store.close()
    print(ans.text)
    t = ans.timings
    print(f"  [{ans.intent} via {ans.route}; gate: {ans.gate or '-'}; {t.get('total_ms', 0):.0f} ms"
          + (f", model {t['llm_ms']:.0f} ms" if "llm_ms" in t else "") + "]")  # fmt: skip
    for c in ans.citations:
        print(f"  cites {c['id']}: {c['text'][:100]}")
    if ans.fact_request:
        print(f"  would ask online (gateway, week 4): {json.dumps(ans.fact_request)}")
    return 0


def cmd_search(args) -> int:
    from kin.memory.embed import Embedder
    from kin.memory.service import Memory
    from kin.memory.store import Store

    _require_models()
    mem = Memory(Store.open(args.db), Embedder())
    try:
        hits = mem.search(args.query, k=args.k, qvec=mem.embedder.embed_one(args.query))
    finally:
        mem.store.close()
    if not hits:
        print("I don't have anything about that.")
    for h in hits:
        when = time.strftime("%a %d %b %H:%M", time.localtime(h["started_at"]))
        sim = "-" if h["vector_similarity"] is None else f"{h['vector_similarity']:.2f}"
        print(f"#e-{h['id']}  {when}  keyword={'yes' if h['keyword_match'] else 'no '}  similarity={sim}")
        print("   ", h["text"][:300])
    return 0


def cmd_asr_test(args) -> int:
    """Compare speech models on the user's own voice. Audio stays in memory only."""
    import numpy as np

    from kin.audio.capture import load_wav
    from kin.perception.asr import Transcriber

    models = args.models.split(",") if args.models else installed_asr_models()
    if not models:
        raise SystemExit("no speech models installed: run `uv run kin fetch-models`")
    if args.wav:
        audio = load_wav(args.wav)
    else:
        import sounddevice as sd

        from kin.audio.capture import find_input_device

        dev = find_input_device(args.device)
        input(f"Press Enter, then speak for up to {args.seconds:.0f} s" + (f': "{args.say}"' if args.say else "") + " ")
        print("recording...")
        rec = sd.rec(int(args.seconds * 16000), samplerate=16000, channels=1, dtype="int16", device=dev)
        sd.wait()
        audio = rec[:, 0].copy()
        del rec
    x = audio.astype(np.float32) / 32768
    rms_db = 20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-9)
    peak_db = 20 * np.log10(np.abs(x).max() + 1e-9)
    print(f"input level: average {rms_db:.0f} dBFS, peak {peak_db:.0f} dBFS", end="  ")
    if peak_db > -1:
        print("(clipping: move back or lower the mic level)")
    elif rms_db < -45:
        print("(very quiet: speak closer or raise the mic level in Windows sound settings)")
    else:
        print("(ok)")

    from kin.perception.textmetrics import wer

    print()
    print(f"{'model':16s} {'time':>7s}  {'error':>6s}  transcript")
    for name in models:
        try:
            asr = Transcriber(threads=2, model=name)
        except Exception as e:  # noqa: BLE001
            print(f"{name:16s} failed to load: {e}")
            continue
        asr.warm_up()
        res = asr.transcribe(audio)
        err = f"{wer(args.say, res.text):6.0%}" if args.say else "     -"
        print(f"{name:16s} {res.decode_ms:5.0f}ms  {err}  {res.text}")
        del asr
    del audio, x
    print()
    print("use one with:  uv run kin run --asr <model>   (or set KIN_ASR=<model>)")
    return 0


def cmd_devices(args) -> int:
    import sounddevice as sd

    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            api = sd.query_hostapis(d["hostapi"])["name"]
            mark = "*" if i == default_in else " "
            print(f"{mark} {i:3d}  {d['name']}  [{api}]")
    return 0


def cmd_doctor(args) -> int:
    ok = True
    problems = verify_models()
    print("models      ", "ok" if not problems else "PROBLEMS")
    for p in problems:
        print("   ", p)
        ok = False
    try:
        from kin.perception.asr import Transcriber

        Transcriber(threads=1).warm_up()
        print("asr engine   ok (sherpa-onnx + pip onnxruntime DLL)")
    except Exception as e:  # noqa: BLE001
        print("asr engine   FAILED:", e)
        ok = False
    try:
        import sounddevice as sd

        from kin.audio.capture import find_input_device

        idx = find_input_device(args.device)
        idx = sd.default.device[0] if idx is None else idx
        sd.check_input_settings(device=idx, samplerate=16000, channels=1, dtype="int16")
        print(f"microphone   ok: {sd.query_devices(idx)['name']} at 16 kHz mono")
    except Exception as e:  # noqa: BLE001
        print("microphone   FAILED:", e)
        ok = False
    try:
        from kin.memory.store import Store

        store = Store.open(args.db)
        st = store.stats()
        store.close()
        print(f"memory       ok: {st['cipher']}, vectors in {st['vector_backend']}, {st['episodes']} episodes")
        if st["vector_backend"] != "sqlite-vec":
            print("             (sqlite-vec did not load; using the numpy fallback)")
    except Exception as e:  # noqa: BLE001
        print("memory       FAILED:", e)
        ok = False
    try:
        from kin.reasoning.llm import LocalLLM

        llm = LocalLLM(threads=get_profile(args.profile).core_threads)
        out = llm.complete("Reply with the single word: ready", max_tokens=4)
        print(f"language     ok: {llm.path.name}, loaded in {llm.load_ms:.0f} ms, replied {out.text!r}")
        llm.unload()
    except Exception as e:  # noqa: BLE001
        print("language     FAILED:", e)
        ok = False
    print("profile     ", dataclasses.asdict(get_profile(args.profile)))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="kin", description="Kin offline voice companion")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, device=True):
        p.add_argument("--profile", choices=["phone", "laptop"], default=None, help="default: $KIN_PROFILE or phone")
        p.add_argument("--db", default=None, help="memory database (default: data/kin.db)")
        p.add_argument("--asr", default=None, help="speech model (see spec/models.toml); default per profile")
        p.add_argument("--calendar", default=None, help="local .ics calendar (default: data/calendar.ics)")
        p.add_argument("--warm", action="store_true", help="load the language model at start-up")
        if device:
            p.add_argument("--device", default=None, help="input device name substring or index")

    p = sub.add_parser("fetch-models")
    p.add_argument("--asr", action="append", help="also fetch this speech model (repeatable, or 'all')")
    p.set_defaults(fn=cmd_fetch_models)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("replay")
    common(p, device=False)
    p.add_argument("wav")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--stay", action="store_true", help="keep the UI up after the replay ends")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("bench")
    common(p)
    p.add_argument("wav", nargs="?", help="omit to measure live mic for --seconds")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--json", help="write results to this file")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("ask", help="ask a question as if spoken after the wake word")
    common(p, device=False)
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("search")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--db", default=None, help="memory database (default: data/kin.db)")
    p.set_defaults(fn=cmd_search)

    p = sub.add_parser("asr-test", help="compare speech models on your own voice (audio is never saved)")
    p.add_argument("--models", help="comma-separated, default: all installed")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--say", help="the sentence you will say; prints word error per model")
    p.add_argument("--wav", help="use a recording instead of the microphone")
    p.add_argument("--device", default=None)
    p.set_defaults(fn=cmd_asr_test)

    p = sub.add_parser("devices")
    p.set_defaults(fn=cmd_devices)

    p = sub.add_parser("doctor")
    common(p)
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
