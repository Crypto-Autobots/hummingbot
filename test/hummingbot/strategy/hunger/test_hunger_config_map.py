import unittest

import hummingbot.strategy.hunger.hunger_config_map as hunger_config_map_module


class HungerConfigMapTests(unittest.TestCase):
    def test_validate_str_for_order_level_should_be_successful(self):
        self.assertIsNone(hunger_config_map_module.validate_str_for_order_level(1))
        self.assertIsNone(hunger_config_map_module.validate_str_for_order_level("1"))
        self.assertIsNone(hunger_config_map_module.validate_str_for_order_level("1%"))
        self.assertIsNone(hunger_config_map_module.validate_str_for_order_level("1.1%"))

    def test_validate_str_for_order_level_should_be_failed(self):
        self.assertEqual(type(hunger_config_map_module.validate_str_for_order_level("1.1")), str)
