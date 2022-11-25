import re
from decimal import Decimal
from enum import Enum
from typing import Union

INTEGER_LEVEL_PATTERN = r"^\d+$"
PERCENTAGE_LEVEL_PATTERN = r"^([0-9]*[.])?[0-9]+%$"


class LevelType(Enum):
    INTEGER = "integer"
    PERCENTAGE = "percentage"

    @classmethod
    def is_valid(cls, value: Union[int, str]) -> bool:
        return type(value) is int or (
            type(value) is str
            and (re.fullmatch(INTEGER_LEVEL_PATTERN, value) or re.fullmatch(PERCENTAGE_LEVEL_PATTERN, value))
        )

    @classmethod
    def from_str(cls, value: Union[int, str]) -> "LevelType":
        if type(value) is int:
            return cls.INTEGER
        if type(value) is str and re.fullmatch(INTEGER_LEVEL_PATTERN, value):
            return cls.INTEGER
        elif type(value) is str and re.fullmatch(PERCENTAGE_LEVEL_PATTERN, value):
            return cls.PERCENTAGE
        raise ValueError(f"Invalid level value: {value}")

    @classmethod
    def to_float(cls, value: Union[int, str]) -> float:
        if type(value) is int:
            return float(value)
        if type(value) is str and re.fullmatch(INTEGER_LEVEL_PATTERN, value):
            return float(value)
        elif type(value) is str and re.fullmatch(PERCENTAGE_LEVEL_PATTERN, value):
            return float(value[:-1])
        raise ValueError(f"Invalid level value: {value}")

    @classmethod
    def to_decimal(cls, value: Union[int, str]) -> Decimal:
        if type(value) is int:
            return Decimal(value)
        if type(value) is str and re.fullmatch(INTEGER_LEVEL_PATTERN, value):
            return Decimal(value)
        elif type(value) is str and re.fullmatch(PERCENTAGE_LEVEL_PATTERN, value):
            return Decimal(value[:-1])
        raise ValueError(f"Invalid level value: {value}")
