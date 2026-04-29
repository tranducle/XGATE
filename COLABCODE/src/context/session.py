"""
Session management for ResearchAgentSystemv14.

This module provides session persistence for interactive mode.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages session persistence for interactive mode.

    Saves and loads conversation history and session state.
    """

    def __init__(
        self,
        session_dir: str = ".sessions",
        session_id: Optional[str] = None
    ):
        """
        Initialize the session manager.

        Args:
            session_dir: Directory to store session files
            session_id: Optional session ID (generated if not provided)
        """
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = self.session_dir / f"{self.session_id}.json"

        # Session state
        self.conversation_history: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}

        # Load existing session if available
        if self.session_file.exists():
            self.load()

        logger.debug(f"SessionManager initialized: {self.session_id}")

    def add_message(
        self,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Role of the message sender ('user', 'assistant', 'system')
            content: Message content
            metadata: Optional additional metadata
        """
        message = {
            'timestamp': datetime.now().isoformat(),
            'role': role,
            'content': content
        }

        if metadata:
            message['metadata'] = metadata

        self.conversation_history.append(message)
        logger.debug(f"Added {role} message to session")

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.add_message('user', content)

    def add_assistant_message(self, content: str, metadata: Dict[str, Any] = None) -> None:
        """Add an assistant message."""
        self.add_message('assistant', content, metadata)

    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.add_message('system', content)

    def get_conversation_history(
        self,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            limit: Optional limit on number of messages

        Returns:
            List of messages
        """
        if limit:
            return self.conversation_history[-limit:]

        return self.conversation_history.copy()

    def get_last_n_messages(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the last n messages.

        Args:
            n: Number of messages to retrieve

        Returns:
            List of the last n messages
        """
        return self.conversation_history[-n:]

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set session metadata.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str = None) -> Any:
        """
        Get session metadata.

        Args:
            key: Optional specific key (None returns all metadata)

        Returns:
            Metadata value or dict of all metadata
        """
        if key:
            return self.metadata.get(key)

        return self.metadata.copy()

    def save(self) -> None:
        """Save session to file."""
        session_data = {
            'session_id': self.session_id,
            'created_at': datetime.now().isoformat(),
            'conversation_history': self.conversation_history,
            'metadata': self.metadata
        }

        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved session to {self.session_file}")

        except Exception as e:
            logger.error(f"Failed to save session: {e}")

    def load(self) -> None:
        """Load session from file."""
        if not self.session_file.exists():
            logger.warning(f"Session file not found: {self.session_file}")
            return

        try:
            with open(self.session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.conversation_history = session_data.get('conversation_history', [])
            self.metadata = session_data.get('metadata', {})

            logger.debug(f"Loaded session from {self.session_file}")

        except Exception as e:
            logger.error(f"Failed to load session: {e}")

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.debug("Cleared conversation history")

    def export_transcript(self, output_path: str = None) -> str:
        """
        Export conversation as a readable transcript.

        Args:
            output_path: Optional output path

        Returns:
            Path to the exported transcript
        """
        if output_path is None:
            output_path = self.session_dir / f"{self.session_id}_transcript.txt"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Session: {self.session_id}\n")
                f.write(f"Exported: {datetime.now().isoformat()}\n")
                f.write("=" * 60 + "\n\n")

                for msg in self.conversation_history:
                    timestamp = msg.get('timestamp', '')
                    role = msg.get('role', '').upper()
                    content = msg.get('content', '')

                    f.write(f"[{timestamp}] {role}:\n")
                    f.write(f"{content}\n")
                    f.write("-" * 60 + "\n\n")

            logger.info(f"Exported transcript to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export transcript: {e}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the session.

        Returns:
            Dict containing session summary
        """
        return {
            'session_id': self.session_id,
            'message_count': len(self.conversation_history),
            'user_messages': sum(1 for m in self.conversation_history if m.get('role') == 'user'),
            'assistant_messages': sum(1 for m in self.conversation_history if m.get('role') == 'assistant'),
            'metadata_keys': list(self.metadata.keys())
        }

    @staticmethod
    def list_sessions(session_dir: str = ".sessions") -> List[str]:
        """
        List all available sessions.

        Args:
            session_dir: Directory containing session files

        Returns:
            List of session IDs
        """
        session_path = Path(session_dir)

        if not session_path.exists():
            return []

        sessions = []
        for file in session_path.glob("*.json"):
            session_id = file.stem
            sessions.append(session_id)

        return sorted(sessions, reverse=True)

    def delete(self) -> None:
        """Delete the session file."""
        if self.session_file.exists():
            self.session_file.unlink()
            logger.info(f"Deleted session: {self.session_id}")
