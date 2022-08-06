from decimal import Decimal
from typing import Optional

from hummingbot.client.config.config_validators import (
    validate_decimal,
    validate_exchange,
    validate_int,
    validate_market_trading_pair,
)
from hummingbot.client.config.config_var import ConfigVar
from hummingbot.client.settings import AllConnectorSettings, required_exchanges


def exchange_on_validated(value: str):
    required_exchanges.add(value)


def maker_trading_pair_prompt():
    exchange = hunger_config_map.get("exchange").value
    example = AllConnectorSettings.get_example_pairs().get(exchange)
    return "Enter the token trading pair you would like to trade on %s%s >>> " % (
        exchange,
        f" (e.g. {example})" if example else "",
    )


# strategy specific validators
def validate_exchange_trading_pair(value: str) -> Optional[str]:
    exchange = hunger_config_map.get("exchange").value
    return validate_market_trading_pair(exchange, value)


# Returns a market prompt that incorporates the connector value set by the user
def market_prompt() -> str:
    exchange = hunger_config_map.get("exchange").value
    return f"Enter the token trading pair on {exchange} >>> "


def order_amount_prompt() -> str:
    base_asset, quote_asset = hunger_config_map.get("market").value.split("-")
    return f"What is the amount of {base_asset} (in {quote_asset}) per order? >>> "


def budget_allocation_prompt() -> str:
    trading_pair = hunger_config_map.get("market").value
    _, quote_asset = trading_pair.split("-")
    return f"What is the budget allocation for {trading_pair} (in {quote_asset})? >>> "


# List of parameters defined by the strategy
hunger_config_map = {
    "strategy": ConfigVar(
        key="strategy",
        prompt="",
        default="hunger",
    ),
    "exchange": ConfigVar(
        key="exchange",
        prompt="Enter the name of the exchange >>> ",
        prompt_on_new=True,
        validator=validate_exchange,
        on_validated=exchange_on_validated,
    ),
    "market": ConfigVar(
        key="market",
        prompt=market_prompt,
        validator=validate_exchange_trading_pair,
        prompt_on_new=True,
    ),
    "order_amount": ConfigVar(
        key="order_amount",
        prompt=order_amount_prompt,
        prompt_on_new=True,
        type_str="decimal",
        validator=lambda v: validate_decimal(
            v,
            min_value=Decimal("0"),
            inclusive=False,
        ),
    ),
    "budget_allocation": ConfigVar(
        key="budget_allocation",
        prompt=budget_allocation_prompt,
        prompt_on_new=True,
        type_str="decimal",
        validator=lambda v: validate_decimal(
            v,
            min_value=Decimal("0"),
            inclusive=False,
        ),
    ),
    "bid_level": ConfigVar(
        key="bid_level",
        prompt="What is the bid level on the order book (1 means best bid)? >>> ",
        prompt_on_new=True,
        type_str="int",
        validator=lambda v: validate_int(v, min_value=1),
        default=3,
    ),
    "ask_level": ConfigVar(
        key="ask_level",
        prompt="What is the ask level on on order book (1 means best ask)? >>> ",
        prompt_on_new=True,
        type_str="int",
        validator=lambda v: validate_int(v, min_value=1),
        default=3,
    ),
    "max_order_age": ConfigVar(
        key="max_order_age",
        prompt="How long do you want to cancel and replace bids and asks with the same price (in seconds)? >>> ",
        type_str="int",
        validator=lambda v: validate_int(v, min_value=0, inclusive=False),
        default=600,
    ),
    "filled_order_delay": ConfigVar(
        key="filled_order_delay",
        prompt="How long do you want to wait before placing the next order if your order gets filled (in seconds)? >>> ",
        type_str="int",
        validator=lambda v: validate_int(v, min_value=0, inclusive=False),
        default=300,
    ),
    "volatility_interval": ConfigVar(
        key="volatility_interval",
        prompt="What is an interval, in second, in which to pick historical mid price data from to calculate market volatility? >>> ",
        type_str="int",
        validator=lambda v: validate_int(v, min_value=1, inclusive=False),
        default=60 * 1,
    ),
    "avg_volatility_period": ConfigVar(
        key="avg_volatility_period",
        prompt="How many interval does it take to calculate average market volatility? >>> ",
        type_str="int",
        validator=lambda v: validate_int(v, min_value=1, inclusive=False),
        default=5,
    ),
    "max_volatility": ConfigVar(
        key="max_volatility",
        prompt="What is the acceptable volatility to start creating new orders or budget reallocation? (Enter 0.1 to indicate 0.1%) >>> ",
        type_str="decimal",
        validator=lambda v: validate_decimal(v, min_value=Decimal("0")),
        default=Decimal("0.1"),
    ),
}
