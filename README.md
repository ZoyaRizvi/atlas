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
