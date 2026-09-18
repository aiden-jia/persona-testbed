import json
import uuid
from datetime import datetime
from pathlib import Path
import config


class ConversationMemory:
    """Persistent conversation session with sliding-window history for API calls."""

    def __init__(
        self,
        session_id: str = None,
        persona_name: str = None,
        prompting_method: str = None,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.persona_name = persona_name or ""
        self.prompting_method = prompting_method or "basic"
        self.messages: list[dict] = []
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self._path = config.SESSIONS_DIR / f"{self.session_id}.json"

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self):
        self.updated_at = datetime.now().isoformat()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, indent=2)

    def export_log(self) -> Path:
        """Copy the session to logs/ as an immutable record."""
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = config.LOGS_DIR / f"{self.session_id}_{stamp}.json"
        payload = self._to_dict()
        payload["exported_at"] = datetime.now().isoformat()
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return log_path

    @classmethod
    def load(cls, session_id: str) -> "ConversationMemory":
        path = config.SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session '{session_id}' not found.")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        mem = cls(
            session_id=data["session_id"],
            persona_name=data.get("persona_name", ""),
            prompting_method=data.get("prompting_method", "basic"),
        )
        mem.messages = data.get("messages", [])
        mem.created_at = data.get("created_at", mem.created_at)
        mem.updated_at = data.get("updated_at", mem.updated_at)
        return mem

    @classmethod
    def list_sessions(cls) -> list[dict]:
        sessions = []
        for path in sorted(
            config.SESSIONS_DIR.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "persona_name": data.get("persona_name", "unknown"),
                        "prompting_method": data.get("prompting_method", "unknown"),
                        "message_count": len(data.get("messages", [])),
                        "updated_at": data.get("updated_at", "")[:16],
                    }
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions

    # ------------------------------------------------------------------ #
    # Message management
    # ------------------------------------------------------------------ #

    def add_message(self, role: str, content: str):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def get_api_history(self, max_messages: int = None) -> list[dict]:
        """Return recent messages as {role, content} dicts for the OpenAI API."""
        limit = max_messages or config.MAX_HISTORY_MESSAGES
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.messages[-limit:]
        ]

    def get_rag_context(self, n_turns: int = 3) -> str:
        """Recent conversation as plain text for embedding-based RAG queries."""
        recent = self.messages[-(n_turns * 2) :]
        lines = []
        for m in recent:
            prefix = "SALESPERSON:" if m["role"] == "user" else "PERSONA:"
            lines.append(f"{prefix} {m['content']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "persona_name": self.persona_name,
            "prompting_method": self.prompting_method,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def __len__(self) -> int:
        return len(self.messages)
