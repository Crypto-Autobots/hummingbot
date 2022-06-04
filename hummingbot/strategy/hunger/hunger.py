#!/usr/bin/env python

import logging
from decimal import Decimal

from hummingbot.core.event.events import OrderType
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.strategy_py_base import StrategyPyBase

hws_logger = None


class HungerStrategy(StrategyPyBase):
    # We use StrategyPyBase to inherit the structure. We also
    # create a logger object before adding a constructor to the class.
    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    def __init__(
        self,
        market_info: MarketTradingPairTuple,
        order_amount: Decimal,
        ask_level: int,
        bid_level: int,
    ):
        super().__init__()
        self._market_info = market_info
        self._exchange_ready = False
        self._order_completed = False
        self.add_markets([market_info.market])
        self._order_amount = order_amount
        self._ask_level = ask_level
        self._bid_level = bid_level

    # After initializing the required variables, we define the tick method.
    # The tick method is the entry point for the strategy.
    def tick(self, timestamp: float):
        if not self._exchange_ready:
            self._exchange_ready = self._market_info.market.ready
            if not self._exchange_ready:
                self.logger().warning(
                    f"{self._market_info.market.name} is not ready. Please wait..."
                )
                return
            else:
                self.logger().warning(
                    f"{self._market_info.market.name} is ready. Trading started"
                )

        if not self._order_completed:
            # The get_mid_price method gets the mid price of the coin and
            # stores it. This method is derived from the MarketTradingPairTuple class.
            best_bid_price = self._market_info.order_book.snapshot[0].price.iloc[
                self._bid_level - 1
            ]
            best_ask_price = self._market_info.order_book.snapshot[1].price.iloc[
                self._ask_level - 1
            ]

            # The buy_with_specific_market method executes the trade for you. This
            # method is derived from the Strategy_base class.
            sell_order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                amount=self._order_amount,
                order_type=OrderType.LIMIT,
                price=Decimal(str(best_ask_price)),
            )
            self.logger().info(f"Submitted sell limit buy order {sell_order_id}")
            buy_order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                amount=self._order_amount,
                order_type=OrderType.LIMIT,
                price=Decimal(str(best_bid_price)),
            )
            self.logger().info(f"Submitted buy limit buy order {buy_order_id}")

            self._order_completed = True

    # Emit a log message when the order completes
    def did_complete_buy_order(self, order_completed_event):
        self.logger().info(
            f"Your limit buy order {order_completed_event.order_id} has been executed"
        )
        self.logger().info(order_completed_event)
