from decimal import Decimal
from unittest import TestCase

from hummingbot.strategy.hunger.types import LevelType


class HungerLevelTypeTests(TestCase):
    def test_is_valid_should_be_successful(self):
        self.assertTrue(LevelType.is_valid(1))
        self.assertTrue(LevelType.is_valid("1"))
        self.assertTrue(LevelType.is_valid("1%"))
        self.assertTrue(LevelType.is_valid("1.1%"))

    def test_is_valid_should_be_failed(self):
        self.assertFalse(LevelType.is_valid(1.1))
        self.assertFalse(LevelType.is_valid("1.1"))
        self.assertFalse(LevelType.is_valid(".1"))
        self.assertFalse(LevelType.is_valid("-1.1%"))


class HungerLevelTypeFromStrTests(TestCase):
    def test_from_str_should_be_successful(self):
        self.assertEqual(LevelType.from_str(1), LevelType.INTEGER)
        self.assertEqual(LevelType.from_str("1"), LevelType.INTEGER)
        self.assertEqual(LevelType.from_str("1%"), LevelType.PERCENTAGE)
        self.assertEqual(LevelType.from_str("1.1%"), LevelType.PERCENTAGE)

    def test_from_str_should_be_failed(self):
        with self.assertRaises(ValueError):
            LevelType.from_str(1.1)
        with self.assertRaises(ValueError):
            LevelType.from_str("1.1")
        with self.assertRaises(ValueError):
            LevelType.from_str(".1")
        with self.assertRaises(ValueError):
            LevelType.from_str("-1.1%")


class HungerLevelTypeToFloatTests(TestCase):
    def test_to_float_should_be_successful(self):
        self.assertEqual(LevelType.to_float(1), 1.0)
        self.assertEqual(LevelType.to_float("1"), 1.0)
        self.assertEqual(LevelType.to_float("1%"), 1.0)
        self.assertEqual(LevelType.to_float("1.1%"), 1.1)

    def test_to_float_should_be_failed(self):
        with self.assertRaises(ValueError):
            LevelType.to_float(1.1)
        with self.assertRaises(ValueError):
            LevelType.to_float("1.1")
        with self.assertRaises(ValueError):
            LevelType.to_float(".1")
        with self.assertRaises(ValueError):
            LevelType.to_float("-1.1%")


class HungerLevelTypeToDecimalTests(TestCase):
    def test_to_decimal_should_be_successful(self):
        self.assertEqual(LevelType.to_decimal(1), Decimal("1"))
        self.assertEqual(LevelType.to_decimal("1"), Decimal("1"))
        self.assertEqual(LevelType.to_decimal("1%"), Decimal("1"))
        self.assertEqual(LevelType.to_decimal("1.1%"), Decimal("1.1"))

    def test_to_decimal_should_be_failed(self):
        with self.assertRaises(ValueError):
            LevelType.to_decimal(1.1)
        with self.assertRaises(ValueError):
            LevelType.to_decimal("1.1")
        with self.assertRaises(ValueError):
            LevelType.to_decimal(".1")
        with self.assertRaises(ValueError):
            LevelType.to_decimal("-1.1%")
