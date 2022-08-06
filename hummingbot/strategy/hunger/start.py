from hummingbot.strategy.hunger import HungerStrategy
from hummingbot.strategy.hunger.hunger_config_map import hunger_config_map as c_map
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple


def start(self):
    try:
        exchange = c_map.get("exchange").value.lower()
        market = c_map.get("market").value
        order_amount = c_map.get("order_amount").value
        budget_allocation = c_map.get("budget_allocation").value
        ask_level = c_map.get("ask_level").value
        bid_level = c_map.get("bid_level").value
        realtime_levels_enabled = c_map.get("realtime_levels_enabled").value
        max_order_age = c_map.get("max_order_age").value
        filled_order_delay = c_map.get("filled_order_delay").value
        volatility_interval = c_map.get("volatility_interval").value
        avg_volatility_period = c_map.get("avg_volatility_period").value
        max_volatility = c_map.get("max_volatility").value

        self._initialize_markets([(exchange, [market])])
        base, quote = market.split("-")
        market_info = MarketTradingPairTuple(self.markets[exchange], market, base, quote)
        self.market_trading_pair_tuples = [market_info]

        self.strategy = HungerStrategy()
        self.strategy.init_params(
            market_info=market_info,
            order_amount=order_amount,
            budget_allocation=budget_allocation,
            ask_level=ask_level,
            bid_level=bid_level,
            realtime_levels_enabled=realtime_levels_enabled,
            max_order_age=max_order_age,
            filled_order_delay=filled_order_delay,
            volatility_interval=volatility_interval,
            avg_volatility_period=avg_volatility_period,
            max_volatility=max_volatility,
        )
    except Exception as exc:
        self.notify(str(exc))
        self.logger().error("Unknown error during initialization.", exc_info=True)
