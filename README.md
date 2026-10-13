# KYC Document Sorter & Segregation Utility

A local Python utility that looks at scanned or photographed customer documents, works out what each one is, and files it into a `KYC`, `NON KYC` or `Unidentified_files` folder. Recognised KYC documents are renamed with their document type.

Classification is done by a vision-language model (`qwen3-vl:4b-instruct` by default) served by [Ollama](https://ollama.com). Images are sent only to the Ollama endpoint you configure, so nothing leaves your machine or network unless you point it at a remote server.

---

## 1. How It Works

For each image in the input folder:

1. **Pick up the file.** Batch mode scans the folder once. Watch mode listens for new files with `watchdog` and waits until each file has stopped growing and can be opened before touching it.
2. **Prepare the image.** A copy is rotated according to its EXIF data, converted to RGB, scaled down so its longest edge is at most 1200 px, and re-encoded as JPEG. The original file is never altered.
3. **Classify it.** The prepared image and a rule-based prompt (`kyc_sorter/detector/prompts.py`) are sent to Ollama. The reply is constrained to a fixed list of categories with a JSON schema, at temperature 0.
4. **Route it.** The original file is copied (default) or moved to the destination, and the result is verified by checking the file size.
5. **Record it.** The outcome is written to a SQLite journal so a later run can skip files that are already done.

Filenames are never used to decide the document type.

### Single-pass and two-pass classification

| Mode | Model calls per image | Behaviour |
|---|---|---|
| Two-pass | 2 | First asks "is this one document or several?" (`kyc_sorter/detector/multi_doc_prompt.py`), then classifies. If the first answer is unclear the file goes to `Unidentified_files`. |
| Single-pass | 1 | The classification prompt also handles the multiple-document case. Faster. |

The shipped `config.yaml` sets `vlm.single_pass_mode: false` (two-pass). The Docker image and the built-in default, used when the setting is absent, are single-pass. Override with `--single-pass`, `--two-pass` or `KYC_SINGLE_PASS_MODE`.

---

## 2. Output Layout and Naming

```
Destination files/
|-- KYC/                   recognised KYC documents, renamed
|   |-- IMG0001_AADHAAR_CARD.jpg
|   |-- IMG0002_PAN_CARD.jpg
|   `-- IMG0002_PAN_CARD_1.jpg      <- name collision, nothing overwritten
|-- NON KYC/               loan agreements, sanction letters, other non-KYC
|   `-- IMG0003.jpg                 images, and images showing several documents
`-- Unidentified_files/    unreadable, ambiguous, failed or errored files
    `-- IMG0004.jpg
```

- **KYC files** are renamed `<original name>_<SUFFIX>.<ext>`. A name that already ends with the suffix is not suffixed again.
- **NON KYC and unidentified files** keep their original name.
- **Collisions** are resolved by appending `_1`, `_2`, and so on. Existing files are never overwritten.
- All KYC documents go into the single `KYC` folder. There are no per-type or per-date subfolders.

### Recognised document types

| Document | Filename suffix |
|---|---|
| Aadhaar Card | `_AADHAAR_CARD` |
| PAN Card | `_PAN_CARD` |
| Voter ID | `_VOTER_ID` |
| Passport | `_PASSPORT` |
| Driving Licence | `_DRIVING_LICENCE` |
| Vehicle Registration Certificate | `_REGISTRATION_CERTIFICATE` |
| Ration Card | `_RATION_CARD` |
| Other Government ID | `_GOVT_ID` |
| Bank Statement | `_BANK_STATEMENT` |
| IT Returns | `_IT_RETURNS` |
| Electricity Bill | `_ELECTRICITY_BILL` |
| Gas Bill | `_GAS_BILL` |
| Phone Bill | `_PHONE_BILL` |
| Municipal Tax Receipt | `_MUNICIPAL_RECEIPT` |
| CKYC Confirmation | `_CKYC_CONFIRMATION` |
| CKYC Declaration | `_CKYC_DECLARATION` |

The model can also answer `NON_KYC`, `MULTIPLE_DOCUMENTS` (routed to `NON KYC`) or `UNIDENTIFIED`. The front and back of the same ID card in one image count as a single document. A loan agreement that was e-signed with Aadhaar is `NON_KYC`, not an Aadhaar card.

---

## 3. Requirements

- Python 3.10 or newer (the Docker image uses 3.11).
- A running Ollama server with the model pulled:
  ```cmd
  ollama pull qwen3-vl:4b-instruct
  ```
- Python packages:
  ```cmd
  pip install -r requirements.txt
  ```
- Tkinter, for the desktop GUI only. It ships with the standard Windows Python installer.

Check that the sorter can reach Ollama and that the model is present:

```cmd
python main.py --check-connection
```

---

## 4. Running

By default the sorter reads from `Source Files/Input` and writes to `Destination files`, both relative to the folder containing `config.yaml`. The input folder is not part of the repository and must exist before you start, so create it or pass `-i`. The destination folders are created as needed.

### Desktop GUI

Double-click `run_sorter.bat`, or run:

```cmd
python main.py --gui
```

Choose the source and destination folders, pick Copy or Move, optionally tick Dry Run, then click **Run Batch Sorter** for a one-off run or **Start Folder Watcher** to keep monitoring the source folder.

### Batch mode

```cmd
python main.py
```

```cmd
python main.py -i "D:\Scans\Incoming" -o "D:\Scans\Sorted"
```

### Watch mode

```cmd
python main.py --watch
```

Add `--process-existing` to also sort files that are already in the folder when the watcher starts. Stop with Ctrl+C.

### Dry run

```cmd
python main.py --dry-run
```

Classifies every file and prints where it would go, without copying or moving anything and without writing to the journal.

### Command-line options

| Option | Purpose |
|---|---|
| `-i`, `--input` | Source folder |
| `-o`, `--output` | Destination folder |
| `--operation copy\|move` | Keep the source file (`copy`, default) or relocate it (`move`) |
| `--dry-run` | Simulate without changing files |
| `-w`, `--watch` | Monitor the source folder continuously |
| `--process-existing` | With `--watch`, also process files already present |
| `-n`, `--limit` | Process at most this many files (batch mode) |
| `--workers` | Number of concurrent requests to Ollama (default 4) |
| `--batch-size` | Files handled per chunk (default 100) |
| `--endpoint`, `--ollama-endpoint` | Ollama URL; a bare host such as `http://localhost:11434` is accepted |
| `--model` | Model name |
| `--timeout` | Request timeout in seconds (default 300) |
| `--keep-alive` | How long Ollama keeps the model loaded, e.g. `24h`, `30m`, `0` |
| `--single-pass`, `--two-pass` | Choose the classification mode |
| `--journal` | Path to the SQLite journal file |
| `--config` | Path to an alternative `config.yaml` |
| `--check-connection` | Test the Ollama connection and model, then exit |
| `--gui` | Launch the desktop interface |

`-c`/`--confidence` and `--retries` are accepted but currently have no effect. See [Known limitations](#9-known-limitations).

---

## 5. Configuration

Settings are resolved in this order, highest priority first:

1. Command-line options
2. Environment variables (`KYC_*`, `OLLAMA_*`)
3. `config.yaml`
4. Built-in defaults

`config.yaml` is looked up in the project root, then in `kyc_sorter/`, then in the current directory. Relative folder paths in it resolve against the folder that contains the file.

### Main settings

| `config.yaml` key | Environment variable | Default | Meaning |
|---|---|---|---|
| `input.directory` | `KYC_INPUT_DIR` | `Source Files/Input` | Source folder |
| `output.directory` | `KYC_OUTPUT_DIR` | `Destination files` | Destination folder |
| `output.operation` | `KYC_OPERATION` | `copy` | `copy` or `move` |
| `processing.dry_run` | `KYC_DRY_RUN` | `false` | Simulate only |
| `processing.max_workers` | `KYC_MAX_WORKERS` | `4` | Concurrent requests |
| `processing.batch_size` | `KYC_BATCH_SIZE` | `100` | Files per chunk |
| `processing.cpu_limit_percent` | `KYC_CPU_LIMIT_PERCENT` | `80` | Pause new work above this CPU load (batch mode) |
| `processing.throttle_cpu_percent` | `KYC_CPU_THROTTLE_PERCENT` | `75` | Drop to one request at a time above this CPU load (batch mode) |
| `processing.journal_path` | `KYC_JOURNAL_PATH` | see below | SQLite journal file |
| `processing.supported_extensions` | `KYC_SUPPORTED_EXTENSIONS` | `.jpg` | Extensions to process; the env var is a comma-separated list |
| `processing.watcher_max_queue` | `KYC_WATCHER_MAX_QUEUE` | `1000` | Pending files allowed before the watcher holds extras back |
| `vlm.endpoint` | `KYC_VLM_ENDPOINT`, `OLLAMA_ENDPOINT`, `OLLAMA_HOST` | `http://localhost:11434/api/chat` | Ollama URL |
| `vlm.model` | `KYC_VLM_MODEL`, `OLLAMA_MODEL` | `qwen3-vl:4b-instruct` | Model name |
| `vlm.timeout` | `KYC_VLM_TIMEOUT`, `OLLAMA_TIMEOUT` | `300` | Request timeout, seconds |
| `vlm.keep_alive` | `KYC_VLM_KEEP_ALIVE`, `OLLAMA_KEEP_ALIVE` | `24h` | How long the model stays loaded |
| `vlm.num_ctx` | `KYC_VLM_NUM_CTX`, `OLLAMA_NUM_CTX` | `8192` | Context window |
| `vlm.single_pass_mode` | `KYC_SINGLE_PASS_MODE` | `false` in `config.yaml` | One model call per image instead of two |
| `vlm.image_max_dimension` | `KYC_IMAGE_MAX_DIMENSION` | `1200` | Longest edge sent to the model, pixels |
| `vlm.image_quality` | `KYC_IMAGE_QUALITY` | `85` | JPEG quality of the image sent to the model |
| `watcher.stability_delay_sec` | `KYC_WATCHER_STABILITY_DELAY` | `1.0` | Gap between size checks on a new file |
| `watcher.settle_timeout_sec` | `KYC_WATCHER_SETTLE_TIMEOUT` | `30.0` | Give up on a file that has not settled |
| `watcher.poll_interval_sec` | none | `0.5` | Delay between readiness checks |
| `watcher.use_polling` | `KYC_WATCHER_USE_POLLING` | `false` (`true` in Docker Compose) | Find new files by re-scanning the folder on a timer instead of OS file events |
| `watcher.polling_interval_sec` | `KYC_WATCHER_POLLING_INTERVAL` | `5.0` | Seconds between scans when polling is on |
| `security.log_sensitive_text` | `KYC_LOG_SENSITIVE_TEXT` | `false` | Turn off masking in logs |

Only `.jpg` files are processed by default (matched case-insensitively, so `.JPG` works). Files with other extensions are left where they are. Temporary and partial files such as `.tmp`, `.part` and `.crdownload`, and names starting with `.` or `~`, are always ignored.

### Startup checks

The sorter refuses to start if:

- the input and output folders are the same, or one is inside the other;
- either folder is under `D:\KYC_Document_sorter` or `D:\KYC_Document_sorter_OCR_VLM`, which are blocked to protect earlier versions of the project;
- a setting is out of range, for example a worker count below 1 or a throttle threshold above the CPU limit.

---

## 6. Reliability Features

### Journal and resume

Every processed file is recorded in a SQLite database. Unless you set a path, it is `kyc_journal.db` in the parent folder of the destination folder (the project root with default settings).

- **Copy mode:** on a re-run, a file that was routed to `KYC` or `NON KYC` is skipped if it is unchanged since it was recorded and its copy is still in the destination. To sort the same files again, delete the journal or point `--journal` at a new file.
- **Unidentified files are retried.** A file that ended in `Unidentified_files` is processed again on the next batch run or `--process-existing` start. In copy mode each retry that fails again leaves another copy there with a `_1`, `_2` suffix.
- **Move mode:** any supported file still present in the source folder is processed.
- A file that was in progress when the program stopped is processed again on the next run.
- A file whose size or modification time has changed is processed again.

### Retries and failures

- A request to Ollama is retried up to 3 times, waiting 1, 2 and 4 seconds, on connection errors and HTTP 500, 502, 503 and 504.
- A file that cannot be read, gets no valid answer, or hits any other error is routed to `Unidentified_files`. One bad file does not stop the run.

### Load control

- In batch mode, files are streamed in chunks so memory use does not grow with folder size.
- In batch mode, CPU load is sampled continuously. Above the throttle threshold only one request runs at a time; above the limit new requests wait, for up to 60 seconds.
- In watch mode, duplicate filesystem events for the same file are ignored and files beyond the queue limit are held back until workers are free.

### Logs

- Per-file results and the end-of-run summary (timings, throughput, counts per document type, CPU, RAM and GPU memory usage) are printed to the console.
- Log messages are also written to `processing_YYYY-MM-DD_HHMMSS.log`. From the command line this goes in a `logs` folder beside the destination folder, or in `KYC_LOG_DIR` if set. The GUI writes to a `logs` folder in the directory it was started from.
- In log messages, 12-digit Aadhaar-style numbers, PAN-format identifiers and 9 to 18 digit numbers are masked unless `security.log_sensitive_text` is `true`.

---

## 7. Docker

The image contains the sorter only. Ollama must be running separately, on the Docker host by default.

1. Copy `.env.example` to `.env` and edit it. `HOST_INPUT_DIR` and `HOST_OUTPUT_DIR` ship with example Windows paths that you need to change.
2. Build the image:
   ```cmd
   docker compose build
   ```
3. Check that the container can reach Ollama:
   ```cmd
   docker compose run --rm kyc-sorter-batch --check-connection
   ```
4. Run a one-off batch:
   ```cmd
   docker compose run --rm kyc-sorter-batch
   ```
   Or start the always-on watcher:
   ```cmd
   docker compose up -d kyc-sorter-watcher
   ```

| Host folder (`.env` variable) | Default without `.env` | Mounted at |
|---|---|---|
| `HOST_INPUT_DIR` | `./data/incoming` | `/data/input` |
| `HOST_OUTPUT_DIR` | `./data/sorted` | `/data/output` |
| `HOST_JOURNAL_DIR` | `./data/journal` | `/data/journal` |
| `HOST_LOG_DIR` | `./data/logs` | `/data/logs` |

The watcher service polls the input folder every 5 seconds (`KYC_WATCHER_USE_POLLING=true`). A Windows or macOS folder mounted into a Linux container does not deliver file-change events, so without polling the watcher starts normally but never notices new files. It only picks up files that arrive after it starts; run the batch service for anything already in the folder.

Inside a container, `localhost` is the container itself. Use `http://host.docker.internal:11434` (the default) for Ollama on the host, or the address of a remote Ollama server. The container runs as a non-root user (UID 10001), so the mounted folders must be writable by it. The containers run in single-pass mode with 2 workers unless `.env` says otherwise; `.env.example` sets 4 workers.

---

## 8. Tests

The tests use `pytest`, which is not listed in `requirements.txt`:

```cmd
pip install pytest
```

```cmd
python -m pytest
```

`pytest.ini` points at `kyc_sorter/tests`.

---

## 9. Known Limitations

These settings are read, validated and in some cases displayed, but the current classification pipeline does not act on them:

- **Confidence threshold** (`processing.confidence_threshold`, `-c`, the GUI slider). The model returns a category only, with no confidence score.
- **Date subfolders** (`output.create_date_folder`, the GUI checkbox). Files always go directly into the three folders.
- **Folder names and suffixes** (`output.kyc_folder`, `output.unknown_folder`, `output.unidentified_folder`, `unknown.*`, `documents.*.filename_suffix`). The names `KYC`, `NON KYC`, `Unidentified_files` and the suffixes in section 2 are fixed in `kyc_sorter/sorter/vlm_router.py`.
- **Retry count and sampling** (`vlm.max_retries`, `--retries`, `vlm.temperature`, `vlm.num_predict`). These are fixed at 3 retries, temperature 0 and 50 tokens.
- **Limit from config** (`processing.limit`, `KYC_LIMIT`). Only the `-n`/`--limit` command-line option limits a batch.
- **`processing.adaptive_workers`** and **`processing.multiple_document_score_margin`**.

Other points to be aware of:

- No CSV report and no companion `.txt` or `.json` files are produced. The GUI's completion dialog still shows a "CSV Report" line, which reads `None`.
- CPU throttling applies to batch mode only, not to watch mode.
- `kyc_sorter/README.md` and `kyc_sorter/deployment/README_DEPLOYMENT.md` still describe the earlier PaddleOCR-based version and do not match the current code.

---

## 10. Project Structure

```
.
|-- main.py                     Root launcher, delegates to kyc_sorter/main.py
|-- config.yaml                 Configuration
|-- requirements.txt            Python dependencies
|-- run_sorter.bat              Windows launcher for the GUI
|-- Dockerfile
|-- docker-compose.yml          Batch and watcher services
|-- .env.example                Template for Docker settings
|-- pytest.ini
`-- kyc_sorter/
    |-- main.py                 Command-line entry point
    |-- gui.py                  Tkinter desktop interface
    |-- detector/
    |   |-- prompts.py          Classification prompt
    |   `-- multi_doc_prompt.py Multiple-document prompt and answer parser
    |-- sorter/
    |   |-- vlm_router.py       Image preparation, Ollama calls, routing, batch execution
    |   |-- batch_processor.py  Batch orchestration, journal integration, summary
    |   |-- watcher.py          Folder watcher
    |   |-- journal.py          SQLite journal
    |   `-- file_router.py      Earlier routing engine, not called by the current pipeline
    |-- utils/
    |   |-- config_loader.py    Config loading, environment overrides, validation
    |   |-- healthcheck.py      Ollama connectivity check
    |   `-- logger.py           Logging with masking
    |-- deployment/
    |   `-- build_exe.py        PyInstaller build script
    `-- tests/
```

`kyc_sorter/` also holds identical copies of `config.yaml`, `requirements.txt` and `run_sorter.bat`. The root `config.yaml` is the one loaded when both exist.
