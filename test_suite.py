"""Unittest test runner for all tests."""

import unittest
import asyncio
from test_storage import test_storage_basic, test_conversation_manager_persistence
from test_integration import test_app_sqlite_integration


class TestSQLitePersistence(unittest.TestCase):
    def test_basic_storage(self):
        test_storage_basic()

    def test_manager_persistence(self):
        asyncio.run(test_conversation_manager_persistence())

    def test_integration(self):
        test_app_sqlite_integration()


if __name__ == "__main__":
    unittest.main()
