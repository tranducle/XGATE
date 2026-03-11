"""
Project state management for ResearchAgentSystemv14.

This module provides project state tracking and logging capabilities.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


logger = logging.getLogger(__name__)


class ProjectState:
    """
    Manages research project state and logging.

    Tracks operations, maintains history, and provides optional
    SDP (Standard Directory Protocol) compliance.
    """

    # SDP directories - enforced by default for research workflow
    SDP_DIRECTORIES = [
        "0_Project_Admin",          # Logs, research diary, metadata
        "1_Strategic_Plan",
        "2_Literature_Review",
        "3_Theoretical_Framework",
        "4_Methodology_Design",
        "5_Experiments_Simulations",
        "6_Analysis_Results",
        "7_Manuscript_Draft",
        "8_Project_Management",
        "9_Presentation"
    ]

    def __init__(
        self,
        project_root: str = ".",
        use_sdp: bool = True,  # Changed to True - enforce SDP by default
        log_file: str = "project_log.json"
    ):
        """
        Initialize project state management.

        Args:
            project_root: Root directory for the project
            use_sdp: Whether to create SDP directory structure
            log_file: Name of the log file
        """
        self.project_root = Path(project_root).resolve()
        self.use_sdp = use_sdp
        self.log_file = self.project_root / log_file

        # Create project root if it doesn't exist
        self.project_root.mkdir(parents=True, exist_ok=True)

        # Optional: Create SDP structure
        if use_sdp:
            self._ensure_sdp_structure()

        # Load existing state
        self.state = self._load_state()

        logger.info(f"ProjectState initialized for {self.project_root}")

    def _ensure_sdp_structure(self) -> None:
        """Create SDP directories if they don't exist."""
        for directory in self.SDP_DIRECTORIES:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.debug(f"Ensured SDP directory: {directory}")

    def _load_state(self) -> Dict[str, Any]:
        """Load existing state from log file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logger.debug(f"Loaded state from {self.log_file}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to load state from {self.log_file}: {e}")

        return {
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat(),
            'operations': [],
            'metadata': {}
        }

    def _save_state(self) -> None:
        """Save state to log file."""
        self.state['modified'] = datetime.now().isoformat()

        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved state to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save state to {self.log_file}: {e}")

    def log_operation(
        self,
        operation_type: str,
        details: Dict[str, Any]
    ) -> None:
        """
        Log an operation to the project state.

        Args:
            operation_type: Type of operation (e.g., 'agent_execution', 'api_call')
            details: Operation details
        """
        operation = {
            'timestamp': datetime.now().isoformat(),
            'type': operation_type,
            'details': details
        }

        self.state['operations'].append(operation)
        self._save_state()

        logger.debug(f"Logged operation: {operation_type}")

    def log_request(self, query: str, optimized_query: str = None) -> None:
        """
        Log a user request.

        Args:
            query: Original user query
            optimized_query: Optionally, the optimized query
        """
        self.log_operation('user_request', {
            'query': query,
            'optimized_query': optimized_query or query
        })

    def log_completion(
        self,
        request_id: str,
        agents_used: List[str],
        status: str,
        result: Dict[str, Any] = None
    ) -> None:
        """
        Log a request completion.

        Args:
            request_id: Request identifier
            agents_used: List of agents that were executed
            status: Completion status
            result: Optional result details
        """
        details = {
            'request_id': request_id,
            'agents': agents_used,
            'status': status
        }

        if result:
            details['result_summary'] = {
                'agents_executed': result.get('agents_executed'),
                'agents_failed': result.get('agents_failed')
            }

        self.log_operation('completion', details)

    def log_error(
        self,
        error_type: str,
        message: str,
        context: Dict[str, Any] = None
    ) -> None:
        """
        Log an error.

        Args:
            error_type: Type of error
            message: Error message
            context: Optional error context
        """
        details = {
            'error_type': error_type,
            'message': message,
            'context': context or {}
        }

        self.log_operation('error', details)

    def get_operation_history(
        self,
        operation_type: str = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get operation history.

        Args:
            operation_type: Filter by operation type (None for all)
            limit: Maximum number of entries to return

        Returns:
            List of operations
        """
        operations = self.state.get('operations', [])

        if operation_type:
            operations = [op for op in operations if op.get('type') == operation_type]

        return operations[-limit:]

    def get_metadata(self, key: str = None) -> Any:
        """
        Get metadata from project state.

        Args:
            key: Optional specific key (None returns all metadata)

        Returns:
            Metadata value or dict of all metadata
        """
        metadata = self.state.get('metadata', {})

        if key:
            return metadata.get(key)

        return metadata

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata in project state.

        Args:
            key: Metadata key
            value: Metadata value
        """
        if 'metadata' not in self.state:
            self.state['metadata'] = {}

        self.state['metadata'][key] = value
        self._save_state()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get project statistics.

        Returns:
            Dict containing various statistics
        """
        operations = self.state.get('operations', [])

        stats = {
            'total_operations': len(operations),
            'created': self.state.get('created'),
            'modified': self.state.get('modified'),
            'operation_types': {}
        }

        # Count operation types
        for op in operations:
            op_type = op.get('type', 'unknown')
            stats['operation_types'][op_type] = stats['operation_types'].get(op_type, 0) + 1

        return stats

    def export_log(self, output_path: str = None) -> str:
        """
        Export project log to a file.

        Args:
            output_path: Optional output path (defaults to project_log_export.json)

        Returns:
            Path to the exported file
        """
        if output_path is None:
            output_path = self.project_root / "project_log_export.json"
        else:
            output_path = Path(output_path)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported log to {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"Failed to export log: {e}")
            raise

    def clear_history(self) -> None:
        """Clear operation history while preserving metadata."""
        self.state['operations'] = []
        self.state['modified'] = datetime.now().isoformat()
        self._save_state()

        logger.info("Cleared operation history")
