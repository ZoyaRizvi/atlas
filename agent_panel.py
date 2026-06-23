"""
atlas/phase4/agent_panel.py

PySide6 widget for the agentic AI panel. Shows:
  - A question input box
  - A live reasoning trace (updates as each tool call happens)
  - A final answer display

The trace panel is the key Phase 4 deliverable: users can see exactly
WHAT the agent decided to do and WHY, before the final answer arrives.
This transparency is essential for trusting and debugging agentic systems.

Run standalone:
    python3 -m atlas.phase4.agent_panel
"""

from __future__ import annotations
import sys

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QTextEdit, QListWidget,
    QListWidgetItem, QSizePolicy,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from atlas.phase4.agent_worker import AgentSession

# Step type -> display prefix mapping
STEP_ICONS = {
    "tool_call":    "🔧 Tool call  →",
    "tool_result":  "📋 Result     →",
    "final_answer": "✅ Answer     →",
}


class AgentPanel(QWidget):
    """
    The agentic AI panel widget.

    DESIGN: two columns once the window is wide enough —
      left: question input + reasoning trace
      right: final answer

    For simplicity we use a vertical layout here that integrates
    cleanly as a tab in the main chat window.
    """

    def __init__(self, document_store=None):
        super().__init__()
        # Inject a document store so the search_notes tool works.
        # Can be set later via set_document_store() after ingestion.
        if document_store is not None:
            from atlas.phase4.agent_tools import set_document_store
            set_document_store(document_store)

        self._active_session: AgentSession | None = None
        self._build_ui()

    def set_document_store(self, store):
        """Call after notes are ingested to enable the search_notes tool."""
        from atlas.phase4.agent_tools import set_document_store
        set_document_store(store)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Header ---
        header = QLabel("Atlas Agent — ask a question that may need tools")
        header.setWordWrap(True)
        layout.addWidget(header)

        # --- Question input ---
        input_bar = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText(
            "e.g. What did I write about KV caching? or What is sqrt(8192)?"
        )
        self.question_input.returnPressed.connect(self._on_ask)
        input_bar.addWidget(self.question_input)

        self.ask_btn = QPushButton("Ask agent")
        self.ask_btn.clicked.connect(self._on_ask)
        input_bar.addWidget(self.ask_btn)
        layout.addLayout(input_bar)

        # --- Reasoning trace ---
        layout.addWidget(QLabel("Reasoning trace:"))
        self.trace_list = QListWidget()
        self.trace_list.setFont(QFont("Menlo", 10))
        self.trace_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.trace_list)

        # --- Final answer ---
        layout.addWidget(QLabel("Final answer:"))
        self.answer_box = QTextEdit()
        self.answer_box.setReadOnly(True)
        self.answer_box.setMaximumHeight(160)
        layout.addWidget(self.answer_box)

    # ------------------------------------------------------------------
    # Agent interaction
    # ------------------------------------------------------------------

    def _on_ask(self):
        question = self.question_input.text().strip()
        if not question or self._active_session is not None:
            return

        self.trace_list.clear()
        self.answer_box.setPlainText("")
        self._set_input_enabled(False)
        self._add_trace_item("thinking", "Agent is thinking...")

        session = AgentSession(question)
        session.step_occurred.connect(self._on_step)
        session.finished.connect(self._on_finished)
        session.error_occurred.connect(self._on_error)

        self._active_session = session
        session.start()

    def _on_step(self, step_type: str, content: str):
        # Remove the "thinking..." placeholder on first real step
        if (self.trace_list.count() == 1 and
                self.trace_list.item(0).text().startswith("Agent is thinking")):
            self.trace_list.clear()

        self._add_trace_item(step_type, content)

    def _on_finished(self, answer: str, trace: list):
        self.answer_box.setPlainText(answer)
        self._active_session = None
        self._set_input_enabled(True)

    def _on_error(self, message: str):
        self.answer_box.setPlainText("[Error] " + message)
        self._active_session = None
        self._set_input_enabled(True)

    def _add_trace_item(self, step_type: str, content: str):
        prefix = STEP_ICONS.get(step_type, "•")
        text = "%s %s" % (prefix, content[:200])
        item = QListWidgetItem(text)
        if step_type == "tool_call":
            item.setForeground(Qt.GlobalColor.cyan)
        elif step_type == "tool_result":
            item.setForeground(Qt.GlobalColor.green)
        elif step_type == "final_answer":
            item.setForeground(Qt.GlobalColor.yellow)
        self.trace_list.addItem(item)
        self.trace_list.scrollToBottom()

    def _set_input_enabled(self, enabled: bool):
        self.question_input.setEnabled(enabled)
        self.ask_btn.setEnabled(enabled)
        self.ask_btn.setText("Ask agent" if enabled else "Running...")
        if enabled:
            self.question_input.setFocus()


def main():
    app = QApplication(sys.argv)
    panel = AgentPanel()
    panel.setWindowTitle("Atlas — Agent panel (Phase 4 standalone)")
    panel.resize(700, 560)
    panel.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()