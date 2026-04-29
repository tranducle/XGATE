"""
Research Diary Management for ResearchAgentSystemv14.

This module provides automatic research diary tracking that captures
the evolution of research thinking, methodology changes, and key insights.
Unlike project_log which tracks operations, research_diary captures
the intellectual journey of research development.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ResearchDiary:
    """
    Manages the research diary - a living document that tracks research evolution.
    
    Auto-updates on key events:
    - Research idea changes/refinements
    - Title revisions
    - Methodology changes
    - Key literature insights
    - Gap discoveries
    - Direction changes
    """
    
    def __init__(self, project_root: str = "."):
        """
        Initialize research diary.
        
        Args:
            project_root: Root directory for the project
        """
        self.project_root = Path(project_root).resolve()
        self.diary_dir = self.project_root / "0_Project_Admin"
        self.diary_file = self.diary_dir / "research_diary.md"
        self.diary_json = self.diary_dir / "research_diary.json"
        
        # Ensure directory exists
        self.diary_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize diary state
        self.entries: List[Dict[str, Any]] = self._load_entries()
        
        logger.info(f"ResearchDiary initialized at {self.diary_file}")
    
    def _load_entries(self) -> List[Dict[str, Any]]:
        """Load existing diary entries from JSON."""
        if self.diary_json.exists():
            try:
                with open(self.diary_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('entries', [])
            except Exception as e:
                logger.warning(f"Failed to load diary entries: {e}")
        return []
    
    def _save_entries(self) -> None:
        """Save diary entries to JSON and regenerate Markdown."""
        # Save JSON
        data = {
            'last_updated': datetime.now().isoformat(),
            'total_entries': len(self.entries),
            'entries': self.entries
        }
        with open(self.diary_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Regenerate Markdown
        self._generate_markdown()
    
    def _generate_markdown(self) -> None:
        """Generate human-readable Markdown diary from entries."""
        lines = [
            "# 📔 Research Diary",
            "",
            f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
            "---",
            "",
        ]
        
        if not self.entries:
            lines.append("*No entries yet. The diary will be populated automatically as research evolves.*")
        else:
            # Group by date
            current_date = None
            for entry in reversed(self.entries):  # Most recent first
                entry_date = entry.get('timestamp', '')[:10]
                if entry_date != current_date:
                    current_date = entry_date
                    lines.append(f"## {current_date}")
                    lines.append("")
                
                entry_type = entry.get('type', 'note')
                emoji = self._get_emoji(entry_type)
                timestamp = entry.get('timestamp', '')
                time_only = timestamp[11:16] if len(timestamp) > 16 else ''
                
                lines.append(f"### {emoji} [{time_only}] {entry.get('title', 'Untitled')}")
                lines.append("")
                
                if entry.get('content'):
                    lines.append(entry['content'])
                    lines.append("")
                
                if entry.get('details'):
                    for key, value in entry['details'].items():
                        lines.append(f"- **{key}:** {value}")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        with open(self.diary_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def _get_emoji(self, entry_type: str) -> str:
        """Get emoji for entry type."""
        emojis = {
            'idea_evolution': '💡',
            'title_revision': '📝',
            'methodology_change': '🔬',
            'key_insight': '🎯',
            'literature_finding': '📚',
            'gap_discovery': '🔍',
            'direction_change': '🧭',
            'milestone': '🏆',
            'question': '❓',
            'decision': '⚖️',
            'note': '📌'
        }
        return emojis.get(entry_type, '📌')
    
    def _add_entry(
        self,
        entry_type: str,
        title: str,
        content: str = "",
        details: Dict[str, Any] = None
    ) -> None:
        """Add a new diary entry."""
        entry = {
            'id': len(self.entries) + 1,
            'timestamp': datetime.now().isoformat(),
            'type': entry_type,
            'title': title,
            'content': content,
            'details': details or {}
        }
        self.entries.append(entry)
        self._save_entries()
        logger.info(f"Diary entry added: {entry_type} - {title}")
    
    # === AUTO-UPDATE METHODS (Called by agents automatically) ===
    
    def log_idea_evolution(
        self,
        old_idea: str,
        new_idea: str,
        reasoning: str
    ) -> None:
        """Log when a research idea evolves or refines."""
        self._add_entry(
            entry_type='idea_evolution',
            title=f"Idea Refined: {new_idea[:50]}...",
            content=reasoning,
            details={
                'Previous Idea': old_idea,
                'New Idea': new_idea
            }
        )
    
    def log_title_revision(
        self,
        old_title: str,
        new_title: str,
        justification: str
    ) -> None:
        """Log when research title changes."""
        self._add_entry(
            entry_type='title_revision',
            title=f"Title Changed",
            content=justification,
            details={
                'Previous Title': old_title,
                'New Title': new_title
            }
        )
    
    def log_methodology_change(
        self,
        what_changed: str,
        why: str,
        impact: str = ""
    ) -> None:
        """Log methodology or approach changes."""
        self._add_entry(
            entry_type='methodology_change',
            title=f"Methodology Update: {what_changed[:40]}",
            content=why,
            details={
                'Change': what_changed,
                'Impact': impact
            } if impact else {'Change': what_changed}
        )
    
    def log_key_insight(
        self,
        insight: str,
        source: str = "",
        implications: str = ""
    ) -> None:
        """Log a key insight from literature or analysis."""
        self._add_entry(
            entry_type='key_insight',
            title=f"Key Insight",
            content=insight,
            details={
                'Source': source,
                'Implications': implications
            } if source else {}
        )
    
    def log_literature_finding(
        self,
        finding: str,
        papers: List[str] = None,
        relevance: str = ""
    ) -> None:
        """Log important literature findings."""
        details = {}
        if papers:
            details['Key Papers'] = ', '.join(papers[:3])
        if relevance:
            details['Relevance'] = relevance
        
        self._add_entry(
            entry_type='literature_finding',
            title=f"Literature Finding",
            content=finding,
            details=details
        )
    
    def log_gap_discovery(
        self,
        gap: str,
        evidence: str = "",
        opportunity: str = ""
    ) -> None:
        """Log research gap discoveries."""
        self._add_entry(
            entry_type='gap_discovery',
            title=f"Gap Identified: {gap[:40]}",
            content=gap,
            details={
                'Evidence': evidence,
                'Opportunity': opportunity
            } if evidence else {}
        )
    
    def log_direction_change(
        self,
        old_direction: str,
        new_direction: str,
        reason: str
    ) -> None:
        """Log major direction changes in research."""
        self._add_entry(
            entry_type='direction_change',
            title=f"Direction Changed",
            content=reason,
            details={
                'Previous Direction': old_direction,
                'New Direction': new_direction
            }
        )
    
    def log_decision(
        self,
        decision: str,
        alternatives_considered: List[str] = None,
        rationale: str = ""
    ) -> None:
        """Log important research decisions."""
        details = {'Rationale': rationale} if rationale else {}
        if alternatives_considered:
            details['Alternatives'] = ', '.join(alternatives_considered)
        
        self._add_entry(
            entry_type='decision',
            title=f"Decision: {decision[:50]}",
            content=decision,
            details=details
        )
    
    def log_milestone(
        self,
        milestone: str,
        achievement: str = ""
    ) -> None:
        """Log research milestones."""
        self._add_entry(
            entry_type='milestone',
            title=f"Milestone: {milestone}",
            content=achievement,
            details={}
        )
    
    def log_question(
        self,
        question: str,
        context: str = ""
    ) -> None:
        """Log open questions that need resolution."""
        self._add_entry(
            entry_type='question',
            title=f"Open Question",
            content=question,
            details={'Context': context} if context else {}
        )
    
    def get_recent_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent diary entries."""
        return list(reversed(self.entries[-limit:]))
    
    def get_entries_by_type(self, entry_type: str) -> List[Dict[str, Any]]:
        """Get all entries of a specific type."""
        return [e for e in self.entries if e.get('type') == entry_type]
    
    def export_diary(self) -> str:
        """Return the path to the Markdown diary file."""
        return str(self.diary_file)


# Convenience function for quick diary access
def get_diary(project_root: str = ".") -> ResearchDiary:
    """Get or create a ResearchDiary instance."""
    return ResearchDiary(project_root)
