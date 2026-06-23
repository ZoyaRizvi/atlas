from __future__ import annotations

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QScrollArea, QLineEdit, QPushButton, QLabel, QComboBox, QFrame,
    QSizePolicy, QFileDialog, QCheckBox,
)
from PySide6.QtCore import Qt, QTimer, QObject, QThread, Signal
from PySide6.QtGui import QFont

from grok_client import GrokStreamSession
from qdrant_retrieval import (
    RetrievalError,
    answer_query_with_sources,
    build_qdrant_collection,
    build_retrieval_chain,
    format_sources,
)


AVAILABLE_MODELS = [
    "grok-2",
    "grok-vision",
]

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "atlas_docs"


class MessageBubble(QFrame):

    def __init__(self, role: str, initial_text: str = ""):
        super().__init__()
        self.role = role
        self._full_text = initial_text

        self.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)

        self.label = QLabel(initial_text)
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.label)

        if role == "user":
            self.setStyleSheet(
                "QFrame { background-color: #2b6cb0; border-radius: 10px; } "
                "QLabel { color: white; }"
            )
            self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Minimum)
        elif role == "error":
            self.setStyleSheet(
                "QFrame { background-color: #7a2020; border-radius: 10px; } "
                "QLabel { color: white; }"
            )
        else:
            self.setStyleSheet(
                "QFrame { background-color: #3a3a3a; border-radius: 10px; } "
                "QLabel { color: #e8e8e8; }"
            )

    def append_text(self, text: str):
        self._full_text += text
        self.label.setText(self._full_text)

    def full_text(self) -> str:
        return self._full_text


class FolderIndexerWorker(QObject):
    finished = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, folder_path: str, qdrant_url: str, collection_name: str):
        super().__init__()
        self.folder_path = folder_path
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

    def run(self):
        try:
            build_qdrant_collection(
                folder_path=self.folder_path,
                collection_name=self.collection_name,
                qdrant_url=self.qdrant_url,
            )
            self.finished.emit(
                f"Imported folder into Qdrant collection '{self.collection_name}'."
            )
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class RetrievalWorker(QObject):
    finished = Signal(str, list)
    error_occurred = Signal(str)

    def __init__(self, chain, query: str):
        super().__init__()
        self.chain = chain
        self.query = query

    def run(self):
        try:
            answer, sources = answer_query_with_sources(self.chain, self.query)
            self.finished.emit(answer, sources)
        except Exception as exc:
            self.error_occurred.emit(str(exc))


class ChatWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Atlas — AI research assistant")
        self.resize(760, 720)

        self.history: list[dict] = []
        self._active_session: GrokStreamSession | None = None
        self._active_bubble: MessageBubble | None = None
        self._retrieval_chain = None
        self._retrieval_ready = False
        self._retrieval_collection = DEFAULT_COLLECTION_NAME
        self._qdrant_url = DEFAULT_QDRANT_URL
        self._index_thread: QThread | None = None
        self._retrieval_thread: QThread | None = None

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("Model:"))
        self.model_picker = QComboBox()
        self.model_picker.addItems(AVAILABLE_MODELS)
        top_bar.addWidget(self.model_picker)

        top_bar.addWidget(QLabel("Qdrant URL:"))
        self.qdrant_url_input = QLineEdit(DEFAULT_QDRANT_URL)
        self.qdrant_url_input.setMaximumWidth(220)
        top_bar.addWidget(self.qdrant_url_input)

        self.import_button = QPushButton("Import folder")
        self.import_button.clicked.connect(self.on_import_folder_clicked)
        top_bar.addWidget(self.import_button)

        self.retrieval_toggle = QCheckBox("Use retrieved docs")
        self.retrieval_toggle.setEnabled(False)
        self.retrieval_toggle.toggled.connect(self.on_retrieval_toggled)
        top_bar.addWidget(self.retrieval_toggle)

        top_bar.addStretch()
        root.addLayout(top_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.message_container = QWidget()
        self.message_layout = QVBoxLayout(self.message_container)
        self.message_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.message_layout.addStretch()
        self.scroll_area.setWidget(self.message_container)
        root.addWidget(self.scroll_area)

        input_bar = QHBoxLayout()
        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("Ask Atlas anything...")
        self.input_box.returnPressed.connect(self.on_send_clicked)
        input_bar.addWidget(self.input_box)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.on_send_clicked)
        input_bar.addWidget(self.send_button)
        root.addLayout(input_bar)

        self._add_bubble(
            "assistant",
            "Hi — I'm Atlas, powered by Grok. Set XAI_API_KEY in your .env, "
            "and start by importing a folder to enable retrieval with citations."
        )

    def _add_bubble(self, role: str, text: str = "") -> MessageBubble:
        bubble = MessageBubble(role, text)
        count = self.message_layout.count()
        self.message_layout.insertWidget(count - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def _scroll_to_bottom(self):
        QTimer.singleShot(0, lambda: self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        ))

    def on_import_folder_clicked(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select folder containing PDFs / notes",
            "",
            QFileDialog.Option.ShowDirsOnly,
        )
        if not folder:
            return

        self._qdrant_url = self.qdrant_url_input.text().strip() or DEFAULT_QDRANT_URL
        self._set_input_enabled(False)
        self.import_button.setEnabled(False)
        self.retrieval_toggle.setEnabled(False)

        self._active_bubble = self._add_bubble("assistant", f"Indexing folder: {folder}\n")
        worker = FolderIndexerWorker(folder, self._qdrant_url, self._retrieval_collection)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_index_finished)
        worker.error_occurred.connect(self._on_index_error)
        worker.finished.connect(thread.quit)
        worker.error_occurred.connect(thread.quit)

        self._index_thread = thread
        thread.start()

    def _on_index_finished(self, message: str):
        if self._active_bubble is not None:
            self._active_bubble.append_text(f"\n{message}")
        self._retrieval_ready = True
        self.retrieval_toggle.setEnabled(True)
        self.retrieval_toggle.setChecked(True)
        self._set_input_enabled(True)
        self.import_button.setEnabled(True)
        self._active_bubble = None

    def _on_index_error(self, message: str):
        if self._active_bubble is not None:
            self._active_bubble.append_text(f"\n[Import error] {message}")
            self._active_bubble = None
        self._set_input_enabled(True)
        self.import_button.setEnabled(True)
        self.retrieval_toggle.setEnabled(self._retrieval_ready)

    def on_retrieval_toggled(self, checked: bool):
        if checked and not self._retrieval_ready:
            self.retrieval_toggle.setChecked(False)
            return

        self.retrieval_toggle.setText(
            "Use retrieved docs" if checked else "Use LLM only"
        )

    def on_send_clicked(self):
        text = self.input_box.text().strip()
        if not text:
            return
        if self._active_session is not None or self._retrieval_thread is not None:
            return

        self.input_box.clear()
        self._add_bubble("user", text)
        self.history.append({"role": "user", "content": text})

        if self.retrieval_toggle.isChecked() and self._retrieval_ready:
            self._send_retrieval_query(text)
            return

        self._send_standard_query(text)

    def _send_standard_query(self, text: str):
        self._set_input_enabled(False)
        self._active_bubble = self._add_bubble("assistant", "")

        model = self.model_picker.currentText()
        session = GrokStreamSession(
            messages=self.history,
            model=model,
            system="You are Atlas, a helpful and concise research assistant "
                   "running inside a desktop app.",
            max_tokens=1024,
        )
        session.token_received.connect(self._on_token)
        session.finished.connect(self._on_finished)
        session.error_occurred.connect(self._on_error)

        self._active_session = session
        session.start()

    def _send_retrieval_query(self, text: str):
        self._set_input_enabled(False)
        self._active_bubble = self._add_bubble("assistant", "")

        try:
            if self._retrieval_chain is None:
                self._retrieval_chain = build_retrieval_chain(
                    collection_name=self._retrieval_collection,
                    qdrant_url=self._qdrant_url,
                )
        except RetrievalError as exc:
            self._on_error(str(exc))
            return

        worker = RetrievalWorker(self._retrieval_chain, text)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_retrieval_finished)
        worker.error_occurred.connect(self._on_retrieval_error)
        worker.finished.connect(thread.quit)
        worker.error_occurred.connect(thread.quit)

        self._retrieval_thread = thread
        thread.start()

    def _on_retrieval_finished(self, answer: str, sources: list):
        citation_text = format_sources(sources)
        text = answer.strip()
        if citation_text:
            text = f"{text}\n\nSources:\n{citation_text}"

        if self._active_bubble is not None:
            self._active_bubble.append_text(text)

        self.history.append({"role": "assistant", "content": text})
        self._retrieval_thread = None
        self._active_bubble = None
        self._set_input_enabled(True)

    def _on_retrieval_error(self, message: str):
        if self._active_bubble is not None:
            self._active_bubble.append_text(f"[Retrieval error] {message}")
        self._retrieval_thread = None
        self._active_bubble = None
        self._set_input_enabled(True)

    def _on_token(self, text: str):
        if self._active_bubble is not None:
            self._active_bubble.append_text(text)
            self._scroll_to_bottom()

    def _on_finished(self, full_text: str):
        self.history.append({"role": "assistant", "content": full_text})
        self._active_session = None
        self._active_bubble = None
        self._set_input_enabled(True)

    def _on_error(self, message: str):
        if self._active_bubble is not None:
            self._active_bubble.append_text(f"[Error] {message}")
        self._active_session = None
        self._active_bubble = None
        self._set_input_enabled(True)

    def _set_input_enabled(self, enabled: bool):
        self.input_box.setEnabled(enabled)
        self.send_button.setEnabled(enabled)
        self.import_button.setEnabled(enabled)
        self.send_button.setText("Send" if enabled else "Thinking...")
        if enabled:
            self.input_box.setFocus()

    def closeEvent(self, event):
        if self._active_session is not None:
            self._active_session.stop_and_wait()
        if self._index_thread is not None and self._index_thread.isRunning():
            self._index_thread.quit()
            self._index_thread.wait(1000)
        if self._retrieval_thread is not None and self._retrieval_thread.isRunning():
            self._retrieval_thread.quit()
            self._retrieval_thread.wait(1000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 10))
    window = ChatWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
