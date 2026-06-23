[//]: # (Atlas Research Assistant README - improved)

# Atlas — Local Research Assistant

A desktop research assistant that combines a PySide6 GUI with Qdrant vector search and Grok/XAI LLMs for retrieval-augmented question answering with source citations.

## Features

- Desktop UI built with PySide6 (`run.py` / `chat_window.py`)
- Import and index local documents into Qdrant
- Retrieval + LLM answers with citations
- Supports `.pdf`, `.txt`, `.md`, `.csv`, `.json`

## Requirements

- Python 3.9+
- pip
- Docker (recommended for running Qdrant locally)

Install Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Create a `.env` file in the project root or export the variables in your shell environment.

- `XAI_API_KEY` — required to access Grok/XAI LLM endpoints.
- Optional: `OPENAI_API_KEY` for OpenAI embedding fallbacks.

Example `.env`:

```env
XAI_API_KEY="xai-..."
```

## Running Qdrant (local)

Run Qdrant with Docker (binds to `http://localhost:6333`):

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If you run Qdrant elsewhere, pass the URL in the UI or CLI where prompted.

## Launch the App (GUI)

Start the desktop app:

```bash
python3 run.py
```

The window will open; use the **Import folder** button to index documents and then ask questions in the input box.

## CLI Tests / Examples

- Index and interact with sample documents (quick test):

```bash
python3 test_grok_retrieval.py sample_docs
```

- Another CLI helper (legacy structure):

```bash
python3 test_rag_cli.py <notes_folder>
```

## Development notes

- Main UI: `chat_window.py` — constructs UI and threads for streaming LLM responses.
- Client wrapper: `grok_client.py` — streaming Grok/XAI session handling.
- Retrieval logic and Qdrant integration: `qdrant_retrieval.py`.

If you rename or move modules, update imports accordingly.

## Troubleshooting

- No GUI appears: verify `PySide6` is installed and your environment supports GUI apps. On headless servers, the GUI will not display.
- Qdrant connection errors: ensure Qdrant is running and reachable at the configured URL. Example error guidance is included in `qdrant_retrieval.py`.
- Embeddings not available: install `sentence-transformers` or configure `OPENAI_API_KEY` to use OpenAI embeddings.

## Contributing

PRs welcome. Open an issue if you have feature requests or encounter bugs.

## License

MIT-style (add your preferred license file to the repo).
# Atlas Research Assistant

A local research assistant GUI built with PySide6, Qdrant, and Grok.

## Overview

This project provides a desktop app for ingesting documents, indexing them into Qdrant, and asking Grok-style retrieval-augmented questions with source citations.

## Key Features

- PySide6 desktop UI (`run.py`)
- Folder import and document ingestion
- Qdrant vector search backend
- Grok/XAI LLM integration via `XAI_API_KEY`
- Support for `.txt`, `.md`, `.json`, `.csv`, and `.pdf`

## Requirements

- Python 3.9+ (or compatible)
- `pip`
- `docker` (recommended for Qdrant)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in the project root or export the environment variable directly:

```bash
export XAI_API_KEY='xai-...'
```

Or create `.env` with:

```env
XAI_API_KEY='xai-...'
```

## Running Qdrant

The app expects a running Qdrant service at `http://localhost:6333` by default.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

## Run the App

```bash
python3 run.py
```

The GUI should launch. Use the `Import folder` button to load documents and then ask questions.

## Quick Test

A simple CLI test is available:

```bash
python3 test_grok_retrieval.py sample_docs
```

This will:

1. index documents from the provided folder
2. build a retrieval chain
3. enter interactive question mode

## Notes

- The UI is built in `chat_window.py`.
- The retrieval logic is implemented in `qdrant_retrieval.py`.
- If `XAI_API_KEY` is not set, the app will prompt you to configure it.
- Supported document formats are `.pdf`, `.txt`, `.md`, `.csv`, and `.json`.

## Sample Documents

The repository includes `sample_docs/grok_info.txt` and `sample_docs/xai_info.txt` for quick testing.

## Troubleshooting

- If the app fails to start, ensure `PySide6` is installed and your platform supports Qt GUI windows.
- If Qdrant is unreachable, verify the service is running and the URL is correct.
- If embeddings fail, install `sentence-transformers` or set `OPENAI_API_KEY` for OpenAI embeddings.

## Acknowledgments

This project combines local GUI tooling with vector retrieval and LLM-powered answers for research workflows.
