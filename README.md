# Kin, laptop edition: Weeks 1–4

This follows [KIN_ARCHITECTURE.md](KIN_ARCHITECTURE.md), the architecture spec (the earlier plan is in [offlinevoiceagentlaptop.md](offlinevoiceagentlaptop.md)). The demo runs on a laptop, and the design choices keep a Raspberry Pi 4 (4 GB) in mind: see [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).

- **Week 1:** the kin-audio process, the shared-memory ring, the energy gate, voice detection (VAD), the wake word, and kin-core speech-to-text (ASR) feeding a live transcript in the UI, with CPU and RAM measured.
- **Week 2:** transcripts are grouped into ~60 s episodes and stored in an encrypted database with one embedding per episode. Search combines FTS5 keywords with sqlite-vec vectors, and the UI has a Memory tab with a timeline, search and delete.
- **Week 3:** Kin answers. A local language model (Gemma 3 1B on llama.cpp) with GBNF grammars, an intent router, tools for recall, tasks, reminders, calendar and fact requests, and a confidence gate. The recall demo works with Wi-Fi off. Also from the spec: speaker tagging (you or other), people and facts, the nightly consolidator and digests, context fusion (power, idle, meetings), proactive triggers, a conversation buffer, `general.knowledge` and `meta.history`, and Kin ignoring its own voice.
- **Week 4:** Kin answers **by voice** (Piper, local). `kin-gateway` is a fourth process, the only one allowed online, reaching three allowlisted hosts for weather, search and news; every call is in an **egress ledger** with its exact request. kin-audio and kin-core run with no network (a guard on every OS, a network namespace on Linux). The clip's mute switch, status LED and serial link are written; the firmware and hardware are not tested here.

## Run it

```
uv sync                              # Python 3.11 env, CPU-only wheels, no PyTorch
uv run kin fetch-models              # one-time download, sha256 pinned in spec/models.toml
uv run kin doctor [--net]            # models, ASR, microphone, memory, language model, voice, speaker; --net proves the no-network guard
uv run kin run --profile phone       # listen on the default mic, opens the UI; answers are spoken (--no-voice for text only)
uv run kin replay tests/data/ambient.wav --stay   # recorded input; ends up in memory too
uv run kin run --warm --calendar my.ics   # load the model at start-up (for demos); use a local calendar
uv run kin ask "when is the site visit Rohan mentioned?"   # ask from the terminal, same engine
uv run kin ask --online "will it rain tomorrow?"   # this shell may call the gateway code for a live forecast
uv run kin wake-test                 # say "Hey Jarvis" 10 times: how strongly your mic and voice trigger it, and a suggested --wake-threshold
uv run python scripts/wake_check.py  # feed tests/audio/wake.wav to openWakeWord and to Kin's copy, 1280 samples at a time
uv run kin consolidate               # run the nightly housekeeping now (digests, merge facts, expiry)
uv run python tests/spec_report.py --week 4   # the spec's per-ID table with PASS/FAIL (runs the whole suite)
uv run kin search "what did Priya say about the budget"   # search memory from the terminal
uv run kin asr-test --say "I am from Nagpur"   # compare speech models on YOUR voice (audio never saved)
uv run kin run --asr parakeet-0.6b   # pick a speech model (or set KIN_ASR)
uv run kin bench tests/data/command.wav    # headless: transcripts, latency, CPU/RAM table
uv run kin bench --seconds 30        # idle-listening cost on the live mic
uv run kin devices                   # pick a mic with --device "<name part>"
```

The UI runs at `http://127.0.0.1:8765/?token=…`. The token changes every session and is printed at start-up.

- **Live tab:** the transcript, listening state and CPU/RAM. Each line shows the episode it was saved to (`#e-12`). Say **"Hey Jarvis"** and then a question or request, or type one into **Ask Kin**. Answers appear as cards showing:
  - which memories or calendar entries back them (click one to open that episode);
  - how the question was routed and what the gate decided;
  - the timings;
  - for weather, web or news questions, the exact request that went online, and which host answered.

  The header has a **Mute** button, a **speak answers** checkbox and a state pill (listening, capturing, thinking, fetching, muted). The right-hand panel shows the voice reply, an **Enrol my voice** button (about 10 s of speech, then each line is tagged `you` or `other`), and whether kin-core's network is blocked.
- **Egress tab:** every online call with the exact request Kin sent, its host, status and bytes. Nothing else leaves the laptop.
- **Tasks tab:** things Kin heard you commit to ("I'll send it by Friday"), shown with the sentence they came from and waiting for **Add to list** or **Dismiss**. Also your to-do list, upcoming reminders and recently done items. When a reminder fires, a banner and a chime appear.
- **Memory tab:** search, and a timeline grouped by day. Each episode shows whether it is still open, being indexed or indexed, and has its own Delete button. The tab also has **Forget last 10 minutes** and **Wipe everything**, which asks you to type `wipe`.

Memory lives in `data/kin.db`, and its key is in Windows Credential Manager under `kin-memory`. The file cannot be read without that key. Every run also writes per-process CPU and RAM samples to `runs/metrics-*.csv`.

## How Kin answers (week 3)

```
question -> router -> tool(s) -> [language model] -> confidence gate -> answer card
```

- **Router** ([router.py](kin/reasoning/router.py)):
  1. Rules catch formulaic requests such as "remind me in 20 minutes" or "what's on my list".
  2. Otherwise the question is matched against intent centroids: the mean embedding of each intent's examples in [spec/intents.toml](spec/intents.toml).
  3. Only close calls go to the model, through a grammar that allows only intent names. Before that, the choices are narrowed:
     - a question that memory clearly covers goes to recall;
     - personal questions ("my passport number") can never become online lookups;
     - tasks and reminders need explicit wording, never a guess.
  4. A question about something heard that also needs a live fact ("umbrella to the site visit Rohan mentioned") is served as both.
- **Tools and intents:**

  | Intent | Handled by |
  | --- | --- |
  | `recall.query` | hybrid memory search + the `.ics` calendar + the model |
  | `task.add` / `task.list` | rules; the model is asked only for vague due times ("after the meeting") |
  | `reminder.set` | rules + the scheduler tick; reminders missed while Kin was off fire at start-up, marked late |
  | `digest.today` | stored commitments, tasks and episode counts; no model |
  | `fact.*` | builds the typed sheet-4 `FactRequest` (grammar-filled, location rounded to 0.1°, names and codes scrubbed), sends it to kin-gateway, and says the result in words. Offline or with no route, Kin says plainly it can't check that right now |
  | `general.knowledge` | the model alone, offline, with "that's from general knowledge, so it may be out of date". Questions that need today's world ("latest", "today", "price of") go online instead; a topic you have talked about goes to memory first |
  | `meta.history` | rules only: "what was my last question?", "repeat that". Answered from the turns log, no model |
  | `unsupported` | declines, and offers to save a note |

- **Confidence gate** ([gate.py](kin/reasoning/gate.py)). The 1B model is not trusted to cite or detect conflicts itself, so these checks are deterministic:
  - no relevant memory → "I don't have anything about that", without calling the model;
  - every name, weekday, month and number in the answer must appear in a memory or calendar entry, and the entries that contain them become the citations;
  - an answer that fails that check is replaced by the remembered sentence itself;
  - two memories that disagree (same person, different day or time) are read back side by side, with a question.
- **Language model** ([llm.py](kin/reasoning/llm.py)):
  - loaded on first use with `use_mmap=True`, and released after 60 s idle;
  - `--warm` loads it at start-up;
  - it refuses to load with less than 1.5 GB free (Kin then answers in retrieval-only mode);
  - it uses half its threads on battery below 30 %.
- **Conversation** (R7): the last four exchanges go into the prompt and reset after five minutes of silence. A short follow-up ("and where?") is searched together with the question before it, and keeps its topic. Every exchange is kept in the encrypted `turns` table (filtered, seven days) and forgotten with "forget the last N minutes".
- **Commitments** ([commitments.py](kin/reasoning/commitments.py)): first-person promises in ordinary speech become candidates. Forgetting or deleting that speech removes unconfirmed candidates and strips the quote from confirmed tasks.

## Week 4: the online boundary, voice and the clip

```
kin-core (no network) --typed FactRequest, local IPC--> kin-gateway --HTTPS, 3 hosts--> Open-Meteo / search / BBC RSS
                      <--FactResponse + LedgerEntry----                (kin-core stores the ledger row; the UI shows it)
```

- **Gateway** ([kin/gateway/](kin/gateway/)): re-validates the request (unknown fields are refused), scrubs names from the `people` table, phone numbers, e-mail addresses and digit runs from a search, allows only HTTPS to `api.open-meteo.com`, `feeds.bbci.co.uk` and the search host, follows no redirects, caps a request at 2 KB and a response at 16 KB, times out at 3 s, caches for 15 minutes, and returns a ledger entry for every call, including failures and cache hits.
- **Search is off until you configure it**, because it needs a key or your own server: set `KIN_SEARCH_URL` (a SearXNG instance) or `KIN_BRAVE_KEY`. Weather and news need nothing.
- **Local IPC** ([ipc.py](kin/ipc.py)): a named pipe on Windows, a Unix socket in a private directory on Linux, with a per-session key.
- **Network isolation** ([guard.py](kin/guard.py), [docs/NETWORK.md](docs/NETWORK.md)): kin-audio and kin-core cannot open IPv4/IPv6 sockets or resolve names (proved live by `kin doctor --net` and by each process reporting it at start-up). On Linux they also move into an empty network namespace. `.importlinter` fails if anything but `kin.gateway` imports an HTTP library.
- **Voice** ([voice.py](kin/output/voice.py)): Piper speaks answers to voice commands (and reminders); typed questions speak only with the checkbox. While Kin speaks, what the microphone hears is tagged `kin` and never stored or answered; a new wake word cuts the reply off.
- **The clip** ([clip_serial.py](kin/audio/clip_serial.py), [firmware/](firmware/)): the clip's mute switch is reported over USB serial and Kin then stops capturing; only the switch can lift it. The status LED follows Kin's state. If the clip is unplugged Kin switches to the laptop microphone and says so. **The firmware is written but not compiled or flashed, and the serial link is tested against a simulated clip.**

Measured on this laptop (i7-9750H, phone profile):

| | |
| --- | --- |
| Model load | ~1 s |
| Grounded answer, model warm | 0.9–1.3 s |
| First answer, model cold | ~3 s |
| Rule-based intents (tasks, reminders, list, digest) | < 20 ms |
| kin-core memory while thinking and speaking (model, Piper, ASR, embeddings, speaker model all loaded) | ~980 MB private + ~700 MB memory-mapped model (the budget is 1.1 GB private) |
| kin-gateway memory | 34 MB private (design: ~45 MB) |
| Piper voice | real-time factor ~0.04 once warm (a 5 s sentence in 0.2 s); the first sentence after loading takes ~2 s |
| Forecast lookup through the gateway | ~0.7 s |
| kin-core memory after the 60 s idle release | ~330 MB |

## Speech models

The model is picked per profile in [spec/models.toml](spec/models.toml). `--asr NAME` or `KIN_ASR=NAME` overrides it. The UI header shows which model is running.

| Model | Error rate, clean / noisy | Indian names right | Time per 4 s sentence | kin-core RAM | Use |
| --- | --- | --- | --- | --- | --- |
| `moonshine-tiny` | 21.5 % / 25.9 % | 22 / 68 | ~100 ms | ~250 MB | Week 1 model, kept for comparison |
| **`moonshine-base`** | **14.1 % / 15.5 %** | **35 / 68** | **~200 ms** | **~330 MB** | **phone default** |
| `parakeet-0.6b` | 11.1 % / 12.2 % | 41 / 68 | ~400 ms (laptop), ~750 ms (phone) | ~830 MB | laptop default, best accuracy for its speed |
| `whisper-small` | 11.1 % / 11.2 % | 43 / 68 | ~1.9 s | ~800 MB | slow, but Whisper is known to handle accents well |

The scores come from [tests/data/asr_eval](tests/data/asr_eval): 12 sentences with Indian place and person names, in two synthetic US voices, clean and with added room noise. No Indian-English synthetic voice is installed here, so none of this measures your accent. Run `kin asr-test` to measure that. `uv run kin fetch-models` downloads only the phone default; add `--asr parakeet-0.6b`, `--asr whisper-small` or `--asr all` for the others.

If a word keeps coming out wrong:
1. Run `uv run kin asr-test --say "your sentence"` and see which model gets it right on your voice.
2. Check the input level it prints. Very quiet input means you should speak closer or raise the mic level in Windows sound settings.
3. A headset mic is usually much clearer than a laptop's built-in array.

## Tests

```
uv run pytest            # about 390 tests, ~7 min, includes the live-mic test and real-model tests
uv run pytest -m "not mic and not models"   # fast unit tests only, no models needed
```

Tests use a throwaway data folder and a fixed test key. They never touch `data/kin.db` or the OS keyring.

| File | What it proves |
| --- | --- |
| `test_ring.py` | Ring order across wraparound, overrun detection, zeroing after decode, sharing across processes |
| `test_gate.py` | Silence stays closed, speech opens, hangover, adapts to room noise, start-up calibration |
| `test_endpoint.py` | Pre-roll, short pauses don't split, long pauses do, blips ignored, max-length cuts lose no audio |
| `test_models.py` | Silero VAD on speech, silence and noise; wake word hits and misses; scores match the openWakeWord reference; Moonshine word error rate on reference audio; wake-phrase stripping |
| `test_detector.py` | kin-audio pipeline in-process: silence costs no VAD, ambient speech vs wake + command, wake arming expires |
| `test_A5_wake.py` | the wake word on every 80 ms chunk (E41): detects the clip, cuts a sentence mid-way into a command, the 1.5 s / 8 s command window, no trigger on ambient speech, detection over recorded babble at 10 and 5 dB SNR |
| `test_core.py` | Commands jump the queue, audio is zeroed after ASR, overruns are reported without crashing |
| `test_asr_models.py` | Model choice order (`--asr`, then `KIN_ASR`, then profile); every listed model pinned; each installed model transcribes; long utterances split at pauses are not lost; decoder-loop guard; accuracy floor for the default on the names set |
| `test_timeparse.py`, `test_commitments.py` | Everyday time phrases ("after work", "by Friday", "in half an hour"), words that are not dates ("we", "may", "4 chairs"), promises vs. other people's, questions and non-actions |
| `test_router.py` | Held-out phrasings route correctly; personal questions never go online; tasks need explicit wording; memory wins close calls; umbrella question is memory + weather |
| `test_confidence_gate.py` | Grounded answers are cited; invented days, names and numbers are rejected; recording dates are not evidence; retrieval thresholds; conflicts; retrieval-only read-back |
| `test_facts_calendar.py` | Sheet-4 `FactRequest` schema (rounding, frozen, no extra fields); only each kind's fields are sent; names and codes scrubbed from search text; `.ics` reading, reloading, bad files |
| `test_engine.py` | Every intent and gate path with a scripted model: grounded, ungrounded, declined, no memory (model not called), low memory, conflict, umbrella, memory-first search, calendar, tasks, vague due dates, digest, unsupported |
| `test_llm.py` | Real Gemma: grammars always yield valid intents and valid sheet-4 requests; idle release; low-memory refusal; battery; real answers on demo memory under 5 s |
| `test_offline.py` | Week-3 exit criterion: with Python networking disabled, recall answers from memory and the weather part degrades gracefully |
| `test_core_reasoning.py` | Voice commands are answered after their transcript; ambient speech is not; heard commitments become candidates; forget removes them; reminders fire once and flag late ones; task requests; idle-release events |
| `test_filters.py` | OTPs (as digits or spoken words) and valid card numbers are removed; times, years, phone numbers and non-card digit runs are kept |
| `test_store.py` | Run against both sqlite-vec and the numpy fallback: no plaintext on disk and a wrong key fails; 60 s / 30 s episode grouping; FTS5 finds exact names and numbers and resists injection; cosine ranking; hybrid evidence; delete, forget, retention and wipe erase text, keywords and vectors; stale vectors are refused |
| `test_memory_models.py` | The real MiniLM embedder retrieves the right episode, including meaning-only questions ("what did I eat at midday" finds the pasta lunch), and scores unrelated questions low |
| `test_core_memory.py` | What kin-core stores: ambient speech as episodes, commands without the wake phrase, wake-only never; codes filtered before the UI and the store; end-of-replay only after indexing; UI requests; recovery of episodes left open by a crash |
| `test_ui.py`, `test_ui_memory.py` | Token on every endpoint, request forwarding, limits, `confirm=wipe`, DNS-rebinding host check, core errors and timeouts |
| `test_pipeline.py` | All real processes: replay end to end, RAM budgets, crash restart under 2 s, a live transcript over a real WebSocket, the memory API through kin-ui, searchable memory that survives a restart and is encrypted on disk |
| `test_privacy.py` | audio, perception, memory and core import no networking modules; kin-audio writes no files; UI on loopback only |
| `test_mic.py` | The real microphone fills the ring at 16 kHz with no overflows |
| `test_w1_w3_ids.py`, `test_w3_gaps.py`, `test_w4_ids.py`, `test_degrade.py` | Named by spec ID (`test_E25_...`, `test_INV7_...`): every component, edge, flow and invariant of weeks 1 to 4, and every row of sheet 8. `tests/spec_report.py` prints the table |
| `test_pi_portability.py` | The Linux-only paths (Unix-socket IPC, network namespace flags, key-file fallback) as far as a Windows machine can check them, and the Pi memory and thread budgets |

## Measured on this laptop

12 logical cores, Windows 11, phone profile.

| | Measured | Design target |
| --- | --- | --- |
| Idle listening, whole machine CPU | 0.6 % (Week 1), 1.8 % with someone talking nearby | < 2 % |
| kin-audio RAM | 73–79 MB | ~70 MB |
| kin-core RAM (Moonshine v2 Base + MiniLM + store) | ~330 MB | ~130 MB idle |
| kin-ui RAM | 45 MB | ~50 MB |
| ASR, 3.9 s command (moonshine-base) | ~210 ms (real-time factor ~0.055) | — |
| End of speech → text on screen | ≤ 220 ms | — |
| Embed one episode | ~3 ms | — |
| Hybrid search (embed query + FTS5 + sqlite-vec) | 5–6 ms | — |

kin-core is over its idle estimate because both models stay loaded. The design table lists ASR (~120 MB) and embeddings (~50 MB) as separate "active" rows.

## How it maps to the design

- **kin-audio** ([kin/audio/](kin/audio/)): the PortAudio callback only copies into shared memory ([ring.py](kin/audio/ring.py)). A detector loop reads 80 ms chunks: every chunk goes to the wake word, and in 512-sample windows to gate → VAD → endpointing. It sends kin-core sample indices, not audio.
- **kin-core** ([kin/core/process.py](kin/core/process.py)): one "heavy lane" runs every model call, in priority order.
  - Priority order: commands and interactive searches, then ambient speech, then episode embeddings.
  - Each segment is transcribed and then zeroed in the ring.
  - The text is filtered ([filters.py](kin/perception/filters.py)), shown live, and written to memory.
- **Memory** ([kin/memory/](kin/memory/)): the parts are split across a few files.
  - [store.py](kin/memory/store.py): SQLCipher with FTS5 and sqlite-vec (int8, 384-d, cosine), plus the numpy fallback.
  - [embed.py](kin/memory/embed.py): MiniLM int8 run through ONNX Runtime and Rust tokenizers.
  - [segmenter.py](kin/perception/segmenter.py): the episode policy.
  - [keys.py](kin/memory/keys.py): the keyring key.
  - The shared schema is in [spec/schema.sql](spec/schema.sql).
- **Search results report their evidence separately:** whether keywords matched, and a meaning-similarity score. Week 3's confidence gate can then decide whether memory really has an answer.
- **kin-ui** ([kin/ui/](kin/ui/)): FastAPI on 127.0.0.1 with a per-session token and a loopback-only Host check. It never opens the database. Memory requests go to kin-core, which holds the key.
- **Supervisor** ([kin/supervisor.py](kin/supervisor.py)): starts kin-core first, restarts crashed processes within 2 s, and samples CPU/RAM every second.

## Privacy behaviour (sheet 9)

- **Encrypted at rest:** every page is encrypted with SQLCipher, including the write-ahead log (WAL). The key is in the OS keyring. Tests check that no plaintext appears in the files and that a wrong key cannot open them.
- **Deletion is real:** `secure_delete` and FTS5 secure-delete mode overwrite deleted data. "Forget last N minutes" removes exactly those utterances; if an episode loses some text, its vector is dropped and rebuilt. "Wipe everything" deletes the files, destroys the key and starts empty with a new key.
- **Expiry:** transcripts expire after 7 days. The check runs at start-up and hourly.
- **Filtering:** these are removed before text reaches the screen or the disk:
  - full card numbers (13–19 digits passing the Luhn check);
  - any number of 4+ digits that follows "card number", "card ending in", CVV or expiry (partial numbers included);
  - account, Aadhaar and PAN numbers, UPI PINs and IFSC codes;
  - OTPs and PINs.

  Amounts, times, years and phone numbers are kept ("I paid 4500 with my card" is unchanged). When the rules improve, text already stored is re-filtered once at the next start-up and re-indexed. See [filters.py](kin/perception/filters.py).

## Deviations and notes

- **Wake word is "Hey Jarvis" for now.** "Hey Kin" needs a custom openWakeWord model to be trained. To swap it in, change `keyword` and `phrase` in [spec/models.toml](spec/models.toml).
- **No fastembed or openwakeword packages at runtime.** Both models run directly on ONNX Runtime, with the same maths. A parity test covers the wake word. This kept about 85 MB of scipy and scikit-learn out of kin-audio and fastembed's extra dependencies out of kin-core.
- **Embedding model:** it is the `qint8_arm64` build of all-MiniLM-L6-v2, pinned to a fixed Hugging Face commit. That is the file the phone will load, and on this laptop it is also the fastest variant (3.3 ms, cosine 0.988 against the float model).
- **Moonshine v2 has an input limit.** This export returns an empty transcript for audio longer than about 9.5 s. [asr.py](kin/perception/asr.py) splits longer segments at the quietest point, keeping each piece under 8 s, and joins the text.
- **Windows DLL clash.** Windows 11 ships an old `onnxruntime.dll` (1.17) in System32, and sherpa-onnx loaded it and segfaulted. [kin/native.py](kin/native.py) preloads the pip copy first.
- **Frames are 32 ms (512 samples)**, not 30 ms, because that is Silero v5's native window.
- **Process messaging uses `multiprocessing` queues** between kin-audio, kin-core and kin-ui, because they work the same on Windows. Only kin-core ↔ kin-gateway uses the spec's local IPC (named pipe or Unix socket).
- **kin-core has no asyncio.** The spec draws R6 as `asyncio` timers; a once-a-second tick thread does the same job with no event loop, which also keeps sockets out of kin-core.
- **The wake word hears everything.** Until 2026-10-01 it was only fed audio while the energy gate was open, and its 1.3 s context was reset each time the gate reopened; live, that missed most wake words. Now it scores every 80 ms chunk (E41, about 2% of one laptop core; not yet measured on the Pi). It triggers on 2 consecutive chunks at or above the threshold, cuts the sentence being spoken where the phrase starts, and opens a command window: up to 1.5 s of silence after the wake word, closed by end of speech, never longer than 8 s. The UI shows a live wake-score meter with the threshold marked. Noise suppression (SpeexDSP) switches on only where the `speexdsp_ns` package is installed; it has no Windows build, so it is off on the laptop.
- **The speech recogniser is a second chance for the wake word.** A transcript that starts with "hey jarvis", "a jarvis" or just "jarvis" while the mode is ambient is rerouted as a command, logged as `wake_miss`, and tagged "wake miss" in the transcript. If the acoustic detector misses "Hey Jarvis" but the transcript starts with "hey jarvis" (or "a jarvis", as it is often heard), the line is treated as a command. Found in live testing: the detector fired on only 1 or 2 of 10 attempts for one voice, while the transcript was right every time. `uv run kin wake-test` measures the detector on your voice and suggests `--wake-threshold`.
- **Memory pressure sheds before it kills.** At the 1.1 GB private ceiling, kin-core first releases the voice and language models (they reload on demand) and only stops if that is not enough. Measured private memory in kin-core: speech model 235 MB, embedder 38, speaker 39, Piper 106, language model 450, so everything loaded together is close to the ceiling. This build hit 1116 MB in live use before the shedding was added.
- **Moonshine v2 Base, not Tiny,** is the phone default: Tiny misheard Indian names too often (see the table above).
- **A5, P2 and G1's search host are stand-ins the spec allows:** "Hey Jarvis" for "Hey Kin"; a two-voice test set for speaker tagging (see Not yet verified); search needs your own key or server.
- **The plan's early risk is resolved on Windows:** sqlite-vec loads inside SQLCipher. The numpy fallback is still there and is tested with the same suite.
- **Grammars constrain tool calls, not free-text answers.** llama.cpp checks every vocabulary entry against the grammar for each token, and Gemma has 262k entries, so grammar-constrained text ran at about 8 tokens/s here, and a top-k shortcut crashed inside llama.cpp. The approach used:
  - Tool calls (intent choice, fact requests, due times) pre-fill the fixed JSON in the prompt, so the grammar constrains only a few tokens.
  - Answers are generated freely (about 21 tokens/s) and then checked by the confidence gate.
- **llama-cpp-python comes from its prebuilt CPU wheel index** (0.3.19). PyPI only ships source, and this laptop has no C++ compiler. That was one of the plan's early risks.
- **Dates:** everyday phrases use Kin's own rules, and `dateparser` is used only for calendar dates such as "12 October". On its own, `dateparser` read "we" and "may" as dates and took 2 s to import.
- **The phone-profile memory limit applies to private memory.** Windows counts the memory-mapped model file in RSS, and the design budgets that as "mostly mmap".
- **Not yet verified (be honest about these in a demo):**
  - **Raspberry Pi.** Nothing has run on one. The Linux paths are written and unit-tested with mocks only. `llama-cpp-python` has no prebuilt Pi wheel and must be compiled there; see [docs/RASPBERRY_PI.md](docs/RASPBERRY_PI.md).
  - **The clip hardware and firmware** (HW1, HW2, HW3, HW7, E01 to E03): source and a bench checklist only.
  - **Speaker tagging on real people.** It is proven on two synthetic voices (David and Zira): the owner scores 0.94 to 0.97 with themselves, the other voice 0.49 to 0.58. Human voices, rooms and colds will score differently. The threshold adapts to how steady your enrolment is, but tune it on your own voice.
  - **Windows firewall layer** is documented, not applied.
  - **`unshare` network namespace** on Linux is unrun.
- **Cut down from the spec:** recurring calendar events and reminders are read as one-off; heard commitments are confirmed in the UI, not by voice; facts are simple "X said …" claims, not a knowledge graph.
