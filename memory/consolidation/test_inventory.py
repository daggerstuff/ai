import unittest

from ai.memory.consolidation.inventory import InventoryEngine, InventoryItem


class TestInventoryEngine(unittest.TestCase):
    def setUp(self):
        # Use in‑memory engine without persistence
        self.engine = InventoryEngine()

    def test_add_and_get(self):
        item = self.engine.add_item(name="test", metadata={"a": 1})
        self.assertIsInstance(item, InventoryItem)
        fetched = self.engine.get_item(item.id)
        self.assertEqual(fetched.name, "test")
        self.assertEqual(fetched.metadata, {"a": 1})

    def test_update(self):
        item = self.engine.add_item(name="orig")
        updated = self.engine.update_item(item.id, name="new", metadata={"b": 2})
        self.assertEqual(updated.name, "new")
        self.assertEqual(updated.metadata, {"b": 2})

    def test_remove(self):
        item = self.engine.add_item(name="to‑remove")
        self.engine.remove_item(item.id)
        with self.assertRaises(KeyError):
            self.engine.get_item(item.id)

    def test_count_and_list(self):
        self.assertEqual(self.engine.count(), 0)
        self.engine.add_item(name="one")
        self.engine.add_item(name="two")
        self.assertEqual(self.engine.count(), 2)
        names = [i.name for i in self.engine.list_items()]
        self.assertCountEqual(names, ["one", "two"])


if __name__ == "__main__":
    unittest.main()
