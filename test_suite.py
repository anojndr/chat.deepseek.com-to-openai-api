# Copyright (c) 2026 chat.deepseek.com-to-openai-api contributors.
"""Run persistence regression tests."""

from __future__ import annotations

import asyncio
import unittest

from test_branching_and_isolation import TestBranchingAndIsolation
from test_include_sources import (
    ChatCompletionsAppendixEndpointTest,
    IncludeSourcesFlagTest,
    ResponsesApiAppendixEndpointTest,
    SourceAppendixFormattingTest,
    StreamEventsSourceEmissionTest,
)
from test_integration import test_app_sqlite_integration
from test_storage import test_conversation_manager_persistence, test_storage_basic
from test_stream_stall import TestStreamStall
from test_turn_recovery import (
    test_cancelled_stream_drops_session,
    test_deterministic_rejection_does_not_rotate,
    test_failed_empty_conversation_leaves_no_row,
    test_failure_invalidates_stale_prefix_refs,
    test_midstream_failure_recovers_with_full_replay,
    test_moderation_does_not_rotate_accounts,
    test_muted_account_is_quarantined_without_fanout,
    test_ready_persisted_before_stream_finishes,
    test_single_account_retries_empty_then_replays,
    test_stream_moderation_does_not_rotate,
    test_stream_single_account_retries_empty,
    test_upload_moderation_quarantines_token_without_fanout,
)
from test_unified_model import TestUnifiedFileTurns, TestUnifiedModel
from test_user_scenario import TestUserScenario

__all__ = [
    "ChatCompletionsAppendixEndpointTest",
    "IncludeSourcesFlagTest",
    "ResponsesApiAppendixEndpointTest",
    "SourceAppendixFormattingTest",
    "StreamEventsSourceEmissionTest",
    "TestBranchingAndIsolation",
    "TestSQLitePersistence",
    "TestStreamStall",
    "TestUnifiedFileTurns",
    "TestUnifiedModel",
    "TestUserScenario",
]


class TestSQLitePersistence(unittest.TestCase):
    """Exercise SQLite persistence paths."""

    @staticmethod
    def test_basic_storage() -> None:
        """Check basic storage round trip."""
        test_storage_basic()

    @staticmethod
    def test_manager_persistence() -> None:
        """Check manager persistence round trip."""
        asyncio.run(test_conversation_manager_persistence())

    @staticmethod
    def test_integration() -> None:
        """Check app SQLite integration."""
        test_app_sqlite_integration()

    @staticmethod
    def test_midstream_failure_recovery() -> None:
        """Check midstream failure recovery."""
        asyncio.run(test_midstream_failure_recovers_with_full_replay())

    @staticmethod
    def test_cancelled_stream_recovery() -> None:
        """Check cancelled stream recovery."""
        asyncio.run(test_cancelled_stream_drops_session())

    @staticmethod
    def test_ready_persisted_early() -> None:
        """Check ready marker persists early."""
        asyncio.run(test_ready_persisted_before_stream_finishes())

    @staticmethod
    def test_single_account_empty_retry() -> None:
        """Check single account empty retry."""
        asyncio.run(test_single_account_retries_empty_then_replays())

    @staticmethod
    def test_stream_single_account_empty_retry() -> None:
        """Check streaming empty retry."""
        asyncio.run(test_stream_single_account_retries_empty())

    @staticmethod
    def test_empty_failure_leaves_no_row() -> None:
        """Check empty failure leaves no row."""
        asyncio.run(test_failed_empty_conversation_leaves_no_row())

    @staticmethod
    def test_stale_prefix_invalidated() -> None:
        """Check stale prefix invalidation."""
        asyncio.run(test_failure_invalidates_stale_prefix_refs())

    @staticmethod
    def test_moderation_no_rotation() -> None:
        """Check moderation fails fast without rotation."""
        asyncio.run(test_moderation_does_not_rotate_accounts())

    @staticmethod
    def test_muted_quarantine_no_fanout() -> None:
        """Check muted token cools down without fanout."""
        asyncio.run(test_muted_account_is_quarantined_without_fanout())

    @staticmethod
    def test_deterministic_no_rotation() -> None:
        """Check deterministic rejection fails fast."""
        asyncio.run(test_deterministic_rejection_does_not_rotate())

    @staticmethod
    def test_upload_moderation_quarantine() -> None:
        """Check flagged file fails fast with quarantine."""
        asyncio.run(test_upload_moderation_quarantines_token_without_fanout())

    @staticmethod
    def test_stream_moderation_no_rotation() -> None:
        """Check stream moderation fails fast."""
        asyncio.run(test_stream_moderation_does_not_rotate())


if __name__ == "__main__":
    unittest.main()
