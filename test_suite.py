"""Unittest test runner for all tests."""

import unittest
import asyncio
from test_storage import test_storage_basic, test_conversation_manager_persistence
from test_integration import test_app_sqlite_integration
from test_turn_recovery import (
    test_midstream_failure_recovers_with_full_replay,
    test_cancelled_stream_drops_session,
    test_ready_persisted_before_stream_finishes,
    test_single_account_retries_empty_then_replays,
    test_stream_single_account_retries_empty,
    test_failed_empty_conversation_leaves_no_row,
    test_failure_invalidates_stale_prefix_refs,
)
from test_user_scenario import TestUserScenario
from test_branching_and_isolation import TestBranchingAndIsolation
from test_include_sources import (
    SourceAppendixFormattingTest,
    IncludeSourcesFlagTest,
    ChatCompletionsAppendixEndpointTest,
    ResponsesApiAppendixEndpointTest,
    StreamEventsSourceEmissionTest,
)
from test_unified_model import TestUnifiedModel, TestUnifiedFileTurns
from test_stream_stall import TestStreamStall


class TestSQLitePersistence(unittest.TestCase):
    def test_basic_storage(self):
        test_storage_basic()

    def test_manager_persistence(self):
        asyncio.run(test_conversation_manager_persistence())

    def test_integration(self):
        test_app_sqlite_integration()

    def test_midstream_failure_recovery(self):
        asyncio.run(test_midstream_failure_recovers_with_full_replay())

    def test_cancelled_stream_recovery(self):
        asyncio.run(test_cancelled_stream_drops_session())

    def test_ready_persisted_early(self):
        asyncio.run(test_ready_persisted_before_stream_finishes())

    def test_single_account_empty_retry(self):
        asyncio.run(test_single_account_retries_empty_then_replays())

    def test_stream_single_account_empty_retry(self):
        asyncio.run(test_stream_single_account_retries_empty())

    def test_empty_failure_leaves_no_row(self):
        asyncio.run(test_failed_empty_conversation_leaves_no_row())

    def test_stale_prefix_invalidated(self):
        asyncio.run(test_failure_invalidates_stale_prefix_refs())


if __name__ == "__main__":
    unittest.main()
