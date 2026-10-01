# Running Kin on a Raspberry Pi 4 (4 GB)

**Status: not run on a Pi.** Everything below is what the code was written to allow, plus the steps that
should work, and what to measure first. Treat the numbers as targets, not results.

## Why the laptop build should port

| Design choice | Why it helps on a Pi |
|---|---|
| The demo runs in the `phone` profile | 4 threads in kin-core, 2 for speech recognition, and a 1.1 GB private-memory ceiling that stops Kin with a clear error instead of letting the Pi swap |
| Every model is ONNX or GGUF | The same files run on ARM: Moonshine, Silero, openWakeWord, MiniLM (the `qint8_arm64` build is what a phone loads), Piper, WeSpeaker, Gemma 3 1B Q4 |
| Piper, not Windows speech | Works on Linux and ARM. Windows SAPI would not |
| Voice model and language model load on demand and unload | Piper unloads after 5 idle minutes, Gemma after 60 s. On the laptop, kin-core drops from ~980 MB to ~330 MB private |
| The language model is memory-mapped | Its ~700 MB is file-backed pages the OS can drop under pressure |
| The gateway is a separate 34 MB process | Adds little; the socket guard and a network namespace keep the other processes offline |
| Unix-socket IPC, key-file fallback, no Windows-only imports at start-up | The Linux paths exist and are unit-tested with mocks (`tests/test_pi_portability.py`) |

## Expected budget on a 4 GB Pi

| | Estimate | Note |
|---|---|---|
| OS and desktop | 0.4 to 0.8 GB | use Raspberry Pi OS Lite (64-bit) and open the UI from another device to save more |
| kin-audio | 50 MB | |
| kin-gateway | 35 MB | |
| kin-core, idle | ~330 MB private | speech model, embedder, store |
| kin-core, thinking and speaking | ~980 MB private | measured on the laptop; the ceiling is 1.1 GB |
| Language model weights (mmap) | ~700 MB | file-backed |
| **Peak total** | **about 2.3 to 2.8 GB** | leaves headroom on 4 GB; do not run a browser on the Pi at the same time |

CPU is the real question. The Pi 4's Cortex-A72 is several times slower than the laptop's i7. Expect Gemma at a few
tokens per second (a two-sentence answer in roughly 5 to 10 s instead of 1 s) and Moonshine-base decoding about
2 to 4 times slower than real time on this laptop's numbers scaled down. Piper's real-time factor of 0.04 here
should stay well under 1. **Measure these first**: `uv run kin bench tests/data/command.wav` and `uv run kin ask "..."`.

## Set-up (untested)

1. Raspberry Pi OS 64-bit (Bookworm), Python 3.11 (`sudo apt install python3.11 python3.11-venv` if the image has a different default), and:
   ```
   sudo apt install libportaudio2 libsndfile1 build-essential cmake git
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. `uv sync`. Things to expect:
   - `llama-cpp-python`: this project pins the prebuilt CPU index for x86 Windows/Linux, which has **no ARM wheel**, so it compiles from source on the Pi (build tools above, roughly 15 to 30 minutes). Remove the `[tool.uv.sources]` index entry for llama-cpp-python on the Pi if `uv sync` insists on it.
   - `sqlcipher3-wheels`, `sqlite-vec`, `sherpa-onnx`, `onnxruntime`, `piper-tts`, `psutil`, `pyserial` publish aarch64 wheels; if one is missing, `uv sync` will say which.
3. `uv run kin fetch-models` (about 1.5 GB in total; the ARM embedder file is the same one).
4. `uv run kin doctor --net`. On the Pi you want to see `layer: netns+guard`. If it says `guard` only, unprivileged user namespaces are off: `sudo sysctl kernel.unprivileged_userns_clone=1`.
5. Headless key storage: with no desktop keyring, Kin keeps the database key in `data/kin.db.key` (mode 0600) and logs a warning. Use full-disk encryption on the Pi's card to cover that gap, or install `gnome-keyring` and unlock it at login. `KIN_KEY_BACKEND=file` chooses the file on purpose; `KIN_KEY_BACKEND=keyring` forbids it.
6. `uv run kin run --profile phone --no-browser --port 8765`, then open the printed URL from another device (use an SSH tunnel: `ssh -L 8765:127.0.0.1:8765 pi@raspberrypi`, since the UI only listens on the Pi's loopback on purpose).
7. Audio: plug in a USB microphone (or the Kin clip). `uv run kin devices` lists inputs. For speech out, set the Pi's default output (HDMI, headphone jack or USB) in `raspi-config`.

## What to measure on the first Pi run

1. `kin doctor --net`: layer reported.
2. `kin bench tests/data/command.wav`: ASR time and real-time factor, idle CPU, RAM per process against the table above.
3. `kin ask` on a remembered fact: cold and warm answer time.
4. Say a question with the wake word and time it from end of speech to the first word out of the speaker.
5. Leave it listening for 30 minutes with the language model idle, and confirm kin-core falls back to ~330 MB and CPU stays under a few percent.

If the answer is too slow on the Pi, the levers are: a smaller language model (Gemma stays the default because the
gate is tuned to it), `--asr moonshine-tiny` (less accurate), fewer episodes per search, and `--no-voice` for
text-only answers.
