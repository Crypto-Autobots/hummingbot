from hummingbot.strategy.hunger import HungerStrategy
from hummingbot.strategy.hunger.hunger_config_map import hunger_config_map as c_map
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


def start(self):
    exchange = c_map.get("exchange").value.lower()
    market = c_map.get("market").value
    order_amount = c_map.get("order_amount").value
    ask_level = c_map.get("ask_level").value
    bid_level = c_map.get("bid_level").value

    self._initialize_markets([(exchange, [market])])
    base, quote = market.split("-")
    market_info = MarketTradingPairTuple(self.markets[exchange], market, base, quote)
    self.market_trading_pair_tuples = [market_info]

    self.strategy = HungerStrategy(
        market_info=market_info,
        order_amount=order_amount,
        ask_level=ask_level,
        bid_level=bid_level,
    )
