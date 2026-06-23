from __future__ import annotations

import os
from PySide6.QtCore import QObject, QThread, Signal

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class StreamWorker(QObject):
    token_received = Signal(str)
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, messages: list[dict], model: str = "grok-2",
                 system: str = "", max_tokens: int = 1024):
        super().__init__()
        self.messages = messages
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self._client = None

    def run(self):
        if OpenAI is None:
            self.error_occurred.emit(
                "The 'openai' package is not installed. "
                "Run: pip install openai"
            )
            return

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            self.error_occurred.emit(
                "XAI_API_KEY environment variable is not set.\n"
                "Set it with: export XAI_API_KEY='xai-...'"
            )
            return

        try:
            self._client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

            full_text = ""
            system_msgs = []
            if self.system:
                system_msgs = [{"role": "system", "content": self.system}]

            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=system_msgs + self.messages,
                stream=True,
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    self.token_received.emit(text)

            self.finished.emit(full_text)

        except Exception as e:
            self.error_occurred.emit(f"Grok error: {type(e).__name__}: {e}")


class GrokStreamSession:

    def __init__(self, messages: list[dict], model: str = "grok-2",
                 system: str = "", max_tokens: int = 1024):
        self._thread = QThread()
        self._worker = StreamWorker(messages, model, system, max_tokens)
        self._worker.moveToThread(self._thread)

        # Connect the thread's started signal to the worker's run method.
        # This is the standard Qt "worker object" pattern — cleaner than
        # subclassing QThread directly because it keeps networking logic
        # separate from thread lifecycle management.
        self._thread.started.connect(self._worker.run)

        # When the worker finishes (success OR error), stop the thread.
        self._worker.finished.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._thread.quit)

        # Expose the worker's signals directly so callers can connect
        # to session.token_received instead of session._worker.token_received
        self.token_received = self._worker.token_received
        self.finished = self._worker.finished
        self.error_occurred = self._worker.error_occurred

    def start(self):
        """Start the worker thread. Returns immediately (non-blocking)."""
        self._thread.start()

    def stop_and_wait(self, timeout_ms: int = 3000):
        self._thread.quit()
        self._thread.wait(timeout_ms)