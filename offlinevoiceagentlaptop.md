System architecture · Laptop demo build · Python over C++

# Kin, laptop edition

The same companion as the phone design, rebuilt for a first demo on a laptop. Python handles orchestration, storage and the UI. Every heavy step (audio I/O, voice detection, speech recognition, embeddings, the language model, encryption) runs inside compiled C/C++ engines that Python calls. The models, database schema, prompts and grammars are the exact files the phone app will load later.

Focus

Memory & recall (primary) · Productivity (secondary)

Demo host

Windows 11, macOS or Ubuntu laptop, 8–16 GB RAM

Accessory

Same clip: USB mic, hardware mute, LEDs

Runtime

Python 3.11, uv-managed, CPU-only wheels, no PyTorch

Profile

`KIN_PROFILE=phone` caps threads and RAM to phone limits

**≈ 1.2 GB**install: \~300 MB Python env + \~0.85 GB models

**\~300 MB**RAM while listening, all processes

**≤ 1.3 GB**RAM peak while thinking, mostly mmap

**\< 2 %**avg CPU idle-listening

Sheet 1

## System, end to end

The tiers and the online boundary are unchanged from the phone design. What changes is how the parts run: four operating-system processes instead of one Android app, so audio capture can never be starved by the model and the reasoning process can be started with networking switched off at the OS level.

The solid teal box is `kin-core`: perception, memory and reasoning in one process that has no network access. Audio lives in `kin-audio`, and the only process allowed online is `kin-gateway`, which accepts typed fact requests over a local Unix socket.

local data flow crosses the online boundary local model privacy invariant forbidden egress

Sheet 2

## Python on top, C++ underneath

Python never touches audio samples in a loop or runs a model itself. Each row below is a thin Python call into a compiled engine; the Python cost is the interpreter (\~40 MB RAM per process) and the glue, not the compute.

| Stage | Python package | Engine underneath | Model | Model disk | RAM active |
| --- | --- | --- | --- | --- | --- |
| Mic capture | `sounddevice` | PortAudio (C) | — | — | \<5 MB |
| Ring buffer, energy gate | `numpy`, `multiprocessing.shared_memory` | NumPy C kernels | — | — | \~1 MB |
| Voice activity | `onnxruntime` | ONNX Runtime (C++) | Silero VAD v5 | \~2 MB | \~10 MB |
| Wake word | `openwakeword` (ONNX mode) | ONNX Runtime | custom “Hey Kin” | \~3 MB | \~20 MB |
| Speech to text | `sherpa-onnx` (alt: `faster-whisper` on CTranslate2) | sherpa-onnx + ONNX Runtime (C++) | Moonshine Tiny int8 (alt: whisper base.en int8) | \~30–75 MB | \~120 MB |
| Speaker tag | `sherpa-onnx` | same engine | small speaker embedder | \~7–25 MB | \~25 MB |
| Embeddings | `fastembed` | ONNX Runtime + Rust tokenizers | all-MiniLM-L6-v2 int8 | \~23 MB | \~50 MB |
| Language model | `llama-cpp-python` | llama.cpp (C++, AVX2 / Apple Metal) | Gemma 3 1B instruct Q4 GGUF | \~0.7 GB | \~0.85 GB |
| Memory store | `sqlcipher3`, `sqlite-vec` | SQLite + SQLCipher + sqlite-vec (C) | — | +\~30 MB/yr | \~10 MB |
| Voice output | `piper-tts` (fallback: `pyttsx3` OS voices) | Piper on ONNX Runtime | one English voice | \~20–60 MB | \~60 MB |
| Fact Gateway | `httpx`, `pydantic` | OpenSSL | — | — | \~45 MB |
| Local UI | `fastapi`, `uvicorn`, plain HTML + WebSocket | — | — | — | \~50 MB |
| Total | Python env \~300 MB on disk. Listening = kin-audio + kin-core idle + gateway + UI. | ≈ 0.85 GB | 300 / 1,300 MB |  |  |

Leaving PyTorch and `transformers` out of the runtime is the single biggest saving: together they would add 1.5–2.5 GB of wheels and hundreds of MB of RAM. Every model here ships as ONNX or GGUF, which are also the formats the Android build uses.

Sheet 3

## Processes, threads and the GIL

Python's global interpreter lock means one busy Python thread can delay another. The audio callback has a hard deadline every 30 ms, so it gets a process of its own. Inside `kin-core`, the C++ engines release the lock while they compute, so ordinary threads are enough.

| Process | Contains | Talks to | Network | RAM |
| --- | --- | --- | --- | --- |
| `kin-audio` | PortAudio callback, ring buffer, energy gate, VAD, wake word, clip serial status | kin-core via shared memory + a queue of speech events | NONE | \~70 MB |
| `kin-core` | Thread pool: ASR, speaker tag, extractor, embedder, memory writes, router, llama.cpp, consolidator, scheduler (asyncio) | kin-audio, kin-gateway (Unix socket), kin-ui (Unix socket) | NONE | \~130 MB idle, ≤1.1 GB thinking |
| `kin-gateway` | Typed fact requests, allowlist, PII scrub, cache, ledger writes | kin-core, three public hosts | ALLOWLIST | \~45 MB |
| `kin-ui` | FastAPI on 127.0.0.1: timeline, tasks, ledger, mute state, LED mirror | kin-core, the local browser | LOOPBACK | \~50 MB |

### Rules that keep it fast

- The audio callback only copies samples into shared memory. No logging, no allocation, no Python work beyond one `numpy` slice.
- ASR runs on segments in a worker thread; llama.cpp gets `n_threads = physical cores − 2`, so the rest of the laptop stays responsive.
- Only one heavy model call at a time in kin-core, through a single priority queue: command answers first, ambient ASR next, consolidation last.
- The language model is loaded with `use_mmap=True` and released after 60 s idle, the same policy as the phone.

### Phone-parity profile

- `KIN_PROFILE=phone` limits kin-core to 4 threads and fails loudly if resident memory passes 1.1 GB.
- The same GGUF and ONNX files, the same `schema.sql`, prompts and GBNF grammars are read from a shared `spec/` folder.
- Latency and memory numbers from the laptop in this profile become the first estimate for the Android port.
- `KIN_PROFILE=laptop` may swap in a larger model for comparison, but the demo runs in phone profile.

Sheet 4

## The online boundary on a laptop

A laptop is easier to lock down than a phone, because the operating system can take networking away from a single process. The reasoning process is started without a network, so the rule is enforced by the kernel rather than by the code being well behaved.

### What may leave

```
class FactRequest(BaseModel):          # pydantic, frozen
    kind: Literal["weather", "search", "news"]
    lat: float | None = None           # rounded to 0.1°
    lon: float | None = None
    hours: conint(ge=0, le=48) = 24
    q: constr(max_length=80) | None = None   # PII-scrubbed
    topic: Literal["local","tech","business","sport"] | None = None
```

kin-core sends this object as JSON over a Unix socket. The gateway re-validates it, scrubs names from the people table, phone numbers, emails and digit runs from `q`, and refuses anything that fails validation.

### Enforcement layers

- **Linux:** kin-audio and kin-core start inside a network namespace (`unshare -n` or systemd `PrivateNetwork=yes`). They cannot open a socket to anything except the local Unix sockets.
- **macOS / Windows:** a guard at kin-core start-up replaces `socket.socket` with a function that raises, before any other import. A per-executable firewall rule blocks the core interpreter as a second layer.
- **Code:** an `import-linter` contract fails CI if any package other than `kin.gateway` imports `httpx`, `requests`, `urllib` or `socket`.
- **Gateway:** httpx transport rejects hosts outside the allowlist, caps requests at 2 KB and responses at 16 KB, times out at 3 s, and writes every call to `egress_log`.
- **Models:** downloaded once by a separate `kin fetch-models` command with hashes pinned in the manifest. The running app never downloads anything.

Sheet 5

## Demo: one question, both sides of the line

“Hey Kin, should I take an umbrella to the site visit Rohan mentioned?”

1. 0 msLOCAL

   kin-audio detects the wake word and posts a speech event. The clip LED goes solid green and an earcon plays.
2. +700LOCAL

   kin-core transcribes the command. The router's centroid match is close between `recall.query` and `fact.weather`, so it hands off to the language model.
3. +900LOCAL

   Context builder retrieves yesterday's episode where Rohan said “site visit Thursday 4 pm at Butibori”, plus the Thursday entry from the local .ics calendar.
4. +1,600LOCAL

   llama.cpp, constrained by a GBNF grammar, emits `{"kind":"weather","lat":21.0,"lon":79.0,"hours":48}`. It does not include the question, Rohan's name or the transcript.
5. +1,650ONLINE

   kin-gateway calls Open-Meteo. LED blinks amber. The UI's ledger shows the exact payload and 96 bytes out. Response: 70% chance of rain Thursday 3–6 pm.
6. +2,100LOCAL

   The model combines memory and forecast, citing episode `#e-4182`. The confidence gate checks the citation and Piper speaks: “Yes. Rohan said Thursday at 4, and there's a 70% chance of rain then.”

### Proof beats for the judges

- Open a terminal and run `kin doctor --net`: it shows kin-core's network namespace has only the loopback interface, or that its socket guard is active.
- Turn off Wi-Fi and ask again. Memory answers still work; Kin says it cannot check the weather right now.
- Slide the mute switch. The green LED goes dark in hardware and the UI shows “Muted, nothing is being heard.”
- Open the ledger: only the weather call is listed for the whole session.

### Seeding memory for the demo

- Run Kin during a short rehearsal conversation the day before, so the Rohan episode is real, not injected.
- Keep a backup: `kin replay demo.wav` feeds a recorded conversation through the same pipeline (the WAV is a demo input file, not a stored recording).
- Warm the model before going on stage so the first answer does not include load time.

Sheet 6

## Memory and intents

### Write path, in real time

- Transcribed speech is tagged *you* or *other*, filtered for OTPs and card numbers, then grouped into \~60 s episodes with a timestamp.
- Rules plus `dateparser` catch commitments (“I'll send it by Friday”). They become task candidates that the user confirms in the UI or by voice.
- One 384-d int8 vector per episode in `sqlite-vec`; text also indexed with FTS5 for exact names and numbers.

### Consolidation, overnight

- Runs only on AC power and after 20 min of user idle (`psutil`).
- Writes a daily digest, merges duplicate facts, keeps source links.
- Transcripts expire after 7 days; digests and facts after a year unless deleted.
- “Forget the last 10 minutes” is a hard delete of text and vectors.

| Intent | Handled by | Example |
| --- | --- | --- |
| `recall.query` | Retrieval + language model | What did Priya say about the budget? |
| `task.add / task.list` | Rules, model only for unclear dates | Remind me to call the landlord after work |
| `reminder.set` | Rules + asyncio timer | In 20 minutes, tell me to leave |
| `digest.today` | Stored digest, no model at query time | What did I commit to today? |
| `fact.weather / fact.search / fact.news` | Language model + Fact Gateway | Is the Wardha road closed? |
| `unsupported` | Honest decline | Draft my tax return |

Sheet 7

## Mute switch and listening light

### Clip accessory, unchanged

- ESP32-S3 with an I²S MEMS mic appears as a standard USB microphone on Windows, macOS and Linux. `sounddevice` selects it by name.
- The slide switch cuts mic power through a MOSFET. The green LED shares that power line, so it is lit exactly when the mic can hear.
- The same USB cable also exposes a serial port. Firmware reports switch state, and kin-audio sends LED patterns back with `pyserial`.

### Laptop-only fallback

- Without the clip, Kin uses the built-in mic and shows a large on-screen listening indicator plus a tray icon.
- Mute is the OS mic mute plus a UI switch. This is labelled as fallback, because it does not meet the physical-switch requirement.

Sheet 8

## Graceful degradation

| Situation | Detected by | Kin does |
| --- | --- | --- |
| No network | Gateway timeout or no route | Answers the local part, says it cannot check the fact now |
| Nothing in memory | Top retrieval score below threshold | “I don't have anything about that.” |
| Conflicting memories | Cited episodes disagree | States both with times and asks which is right |
| Beyond the model | Intent `unsupported` | Declines plainly, offers to save a note |
| Low memory | `psutil.virtual_memory().available` \< 1.5 GB | Unloads the model and answers in retrieval-only mode, reading the best episode |
| On battery, low charge | `psutil.sensors_battery()` | Halves model threads, postpones consolidation |
| Clip unplugged mid-demo | PortAudio device error | Switches to laptop mic, says so on screen |
| A process crashes | Supervisor (`kin run`) | Restarts it within 2 s; audio stops while kin-audio restarts, and the LED shows it |

Sheet 9

## Privacy invariants

|  | Invariant | Where it is enforced |
| --- | --- | --- |
| NEVER | Raw audio written to disk or sent anywhere | Ring buffer in shared memory, zeroed after decode; kin-audio has no file writes and no network |
| NEVER | Cloud model calls | No LLM SDKs installed; import-linter contract; kernel-level no-network for kin-core on Linux |
| NEVER | Telemetry from libraries | `HF_HUB_OFFLINE=1`, `DO_NOT_TRACK=1`, no analytics packages; models loaded from local paths only |
| ALWAYS | Memory encrypted at rest | SQLCipher; key held by `keyring` (Windows Credential Manager, macOS Keychain, Secret Service) |
| ALWAYS | User can see and delete what is stored | Timeline in the UI, per-episode delete, “forget last N minutes”, full wipe |
| LOGGED | Every online call visible with its payload | `egress_log` table and ledger view |
| ALWAYS | UI reachable only from this laptop | uvicorn bound to 127.0.0.1 with a per-session token |

Sheet 10

## Repository, dependencies and build order

### Package layout

```
kin/
  audio/       capture, ring (shm), gate, vad, wakeword, clip_serial
  perception/  asr, speaker, filters, segmenter, extractor
  memory/      store (sqlcipher + sqlite-vec), embed, consolidate
  reasoning/   router, context, llm, tools, gate, scheduler
  gateway/     server, schema, scrub, clients/, ledger
  ui/          FastAPI app, static/, websocket
  supervisor.py  starts and watches the 4 processes
spec/          shared with Android
  models.toml    names, sha256, context, templates
  schema.sql     prompts/*.txt   grammars/*.gbnf
tests/  evals/  firmware/ (ESP-IDF, C)
```

### Runtime dependencies

```
numpy  sounddevice  onnxruntime  openwakeword
sherpa-onnx  fastembed  llama-cpp-python
sqlcipher3  sqlite-vec  keyring  dateparser
pydantic  httpx  fastapi  uvicorn  pyserial
piper-tts  psutil  icalendar
```

Dev only: `pytest`, `import-linter`, `ruff`, and the llama.cpp conversion scripts. Install with `uv sync`; pin exact versions in `uv.lock`.

```
uv run kin fetch-models      # one-time, hashes checked
uv run kin run --profile phone
uv run kin doctor --net
```

### Build order for the laptop demo

- **Week 1:** kin-audio process, shared-memory ring, VAD, wake word; kin-core ASR to a live transcript in the UI. Measure CPU and RAM.
- **Week 2:** episodes, embeddings, encrypted store, FTS5 + vector search, timeline view.
- **Week 3:** llama-cpp-python with grammars, router, tools, confidence gate. Recall demo works with Wi-Fi off.
- **Week 4:** kin-gateway, ledger, network isolation, clip firmware with switch and LEDs, Piper voice, rehearsal.

Risks to test early: `sqlite-vec` loading as an extension inside `sqlcipher3` on each OS (fallback: store vectors as BLOBs and search with numpy, fine below \~100k episodes); prebuilt `llama-cpp-python` wheels for each laptop's CPU; and PortAudio device names changing when the clip is replugged.