#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Opencode Session Database Access Module

Provides functions to read session data from opencode's SQLite database.
This allows the fracture-point-detector skill to access child session data.
"""

import sqlite3
import json
import os
import sys
import logging
from contextlib import closing
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认路径，可通过环境变量 OPENCODE_DB_PATH 覆盖
DEFAULT_DB_PATH = Path.home() / ".local/share/opencode/opencode.db"
DB_PATH = Path(os.environ.get("OPENCODE_DB_PATH", str(DEFAULT_DB_PATH)))


@dataclass
class ToolCall:
    """Represents a tool call from a session."""
    id: str
    tool: str
    call_id: Optional[str]
    status: str
    input: Dict[str, Any]
    output: Optional[str]
    error: Optional[str]
    time_created: int
    
    @property
    def is_completed(self) -> bool:
        return self.status == "completed"
    
    @property
    def has_error(self) -> bool:
        return self.error is not None or self.status == "error"


@dataclass
class SessionInfo:
    """Basic session information."""
    id: str
    title: str
    parent_id: Optional[str]
    time_created: int
    time_updated: int


@dataclass
class TextPart:
    """Represents a text part (user message or assistant reply)."""
    id: str
    text: str
    time_created: int
    role: Optional[str] = None  # 'user' or 'assistant'


@dataclass
class ReasoningPart:
    """Represents assistant's reasoning process."""
    id: str
    text: str
    time_created: int


@dataclass
class SessionParts:
    """All parts from a session."""
    session_id: str
    user_messages: List[TextPart] = field(default_factory=list)
    assistant_replies: List[TextPart] = field(default_factory=list)
    reasoning: List[ReasoningPart] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    step_markers: List[Dict[str, Any]] = field(default_factory=list)
    other_parts: List[Dict[str, Any]] = field(default_factory=list)


def get_connection() -> sqlite3.Connection:
    """Get database connection."""
    if not DB_PATH.exists():
        raise RuntimeError(f"Database not found at {DB_PATH}")
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        raise RuntimeError(f"Failed to connect to database at {DB_PATH}: {e}") from e


def get_current_session_id() -> Optional[str]:
    """
    Get the current session ID by querying the most recently updated root session.
    
    Returns:
        Session ID string, or None if not found
    """
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id
                FROM session
                WHERE parent_id IS NULL
                ORDER BY time_updated DESC
                LIMIT 1
            """)
            row = cursor.fetchone()

        return row[0] if row else None
    except (sqlite3.Error, RuntimeError):
        return None


def get_session_title(session_id: str) -> Optional[str]:
    """Get session title by ID."""
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("""
                SELECT title FROM session WHERE id = ?
            """, (session_id,))
            row = cursor.fetchone()

        return row[0] if row else None
    except (sqlite3.Error, RuntimeError):
        return None


def rows_to_sessions(rows) -> List[SessionInfo]:
    """Map (id, title, parent_id, time_created, time_updated) rows to SessionInfo."""
    sessions = []
    for row in rows:
        sessions.append(SessionInfo(
            id=row[0],
            title=row[1],
            parent_id=row[2],
            time_created=row[3],
            time_updated=row[4],
        ))
    return sessions


def get_child_sessions(parent_id: str) -> List[SessionInfo]:
    """Get all child sessions of a parent session."""
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id, title, parent_id, time_created, time_updated
                FROM session
                WHERE parent_id = ?
                ORDER BY time_created ASC
            """, (parent_id,))
            sessions = rows_to_sessions(cursor.fetchall())

        return sessions
    except (sqlite3.Error, RuntimeError):
        return []


def list_recent_root_sessions(limit: int = 10) -> List[SessionInfo]:
    """
    List recent root sessions (no parent_id) for agent to identify correct session.
    
    Args:
        limit: Maximum number of sessions to return
        
    Returns:
        List of SessionInfo ordered by time_updated DESC
    """
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id, title, parent_id, time_created, time_updated
                FROM session
                WHERE parent_id IS NULL
                ORDER BY time_updated DESC
                LIMIT ?
            """, (limit,))
            sessions = rows_to_sessions(cursor.fetchall())

        return sessions
    except (sqlite3.Error, RuntimeError):
        return []


def _classify_part(part_id: str, time_created: int, message_id: str,
                   data: dict, message_roles: dict) -> tuple:
    # Returns (category, classified_object) where category is one of:
    #   "tool", "text_user", "text_assistant", "reasoning", "step", "other".
    part_type = data.get("type", "unknown")

    if part_type == "tool":
        state = data.get("state") or {}
        tool_call = ToolCall(
            id=part_id,
            tool=data.get("tool", "unknown"),
            call_id=data.get("callID"),
            status=state.get("status", "unknown"),
            input=state.get("input") or {},
            output=state.get("output"),
            error=state.get("error"),
            time_created=time_created
        )
        return ("tool", tool_call)
    elif part_type == "text":
        text_content = data.get("text", "")
        role = message_roles.get(message_id)
        text_part = TextPart(
            id=part_id,
            text=text_content,
            time_created=time_created,
            role=role
        )
        category = "text_user" if role == "user" else "text_assistant"
        return (category, text_part)
    elif part_type == "reasoning":
        reasoning_part = ReasoningPart(
            id=part_id,
            text=data.get("text", ""),
            time_created=time_created
        )
        return ("reasoning", reasoning_part)
    elif part_type in ("step-start", "step-finish"):
        return ("step", {
            "id": part_id,
            "time_created": time_created,
            "type": part_type
        })
    else:
        return ("other", {
            "id": part_id,
            "time_created": time_created,
            "type": part_type,
            "data": data
        })


def get_session_parts(session_id: str) -> SessionParts:
    """
    Get all parts from a session.

    Uses message.role to distinguish user messages from assistant replies.
    """
    try:
        with closing(get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id, json_extract(data, '$.role') as role
                FROM message
                WHERE session_id = ?
            """, (session_id,))
            message_roles = {row[0]: row[1] for row in cursor.fetchall()}

            cursor = conn.execute("""
                SELECT id, time_created, message_id, data
                FROM part
                WHERE session_id = ?
                ORDER BY time_created ASC
            """, (session_id,))

            parts = SessionParts(session_id=session_id)
            dispatch = {
                "tool": parts.tool_calls.append,
                "text_user": parts.user_messages.append,
                "text_assistant": parts.assistant_replies.append,
                "reasoning": parts.reasoning.append,
                "step": parts.step_markers.append,
                "other": parts.other_parts.append,
            }

            for row in cursor.fetchall():
                data = json.loads(row[3])
                category, obj = _classify_part(
                    row[0], row[1], row[2], data, message_roles,
                )
                append = dispatch.get(category, parts.other_parts.append)
                append(obj)

        return parts
    except (sqlite3.Error, RuntimeError, json.JSONDecodeError):
        return SessionParts(session_id=session_id)


if __name__ == "__main__":
    # Quick test
    logger.info("Testing session_db module...")

    if not DB_PATH.exists():
        logger.warning(f"SKIP: database not found at {DB_PATH}")
        sys.exit(2)

    # Get recent root sessions
    recent = list_recent_root_sessions(limit=5)
    logger.info(f"Recent root sessions: {len(recent)}")
    if not recent:
        logger.warning(f"SKIP: no root sessions found in {DB_PATH}")
        sys.exit(2)

    for i, s in enumerate(recent, 1):
        logger.info(f"  {i}. [{s.id[:12]}...] {s.title}")

    # Test get_session_parts
    parts = get_session_parts(recent[0].id)
    logger.info("First session parts:")
    logger.info(f"  User messages: {len(parts.user_messages)}")
    logger.info(f"  Assistant replies: {len(parts.assistant_replies)}")
    logger.info(f"  Tool calls: {len(parts.tool_calls)}")
    logger.info(f"  Reasoning: {len(parts.reasoning)}")

    logger.info("Module works correctly!")