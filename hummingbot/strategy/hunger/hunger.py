#!/usr/bin/env python

import logging
from decimal import Decimal
from typing import Dict, List

from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.event.events import OrderType, TradeType
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.pure_market_making.pure_market_making import PriceSize, Proposal
from hummingbot.strategy.strategy_py_base import StrategyPyBase

s_decimal_zero = Decimal("0")
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
        max_order_age: int,
        filled_order_delay: int,
    ):
        super().__init__()
        self.add_markets([market_info.market])
        self._market_info = market_info
        self._exchange_ready = False
        # Config map
        self._order_amount = order_amount
        self._ask_level = ask_level
        self._bid_level = bid_level
        self._max_order_age = max_order_age
        self._filled_order_delay = filled_order_delay
        # Global timestamp
        self._create_timestamp = 0
        self._created_timestamp = 0

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

        proposal = None

        if self._create_timestamp < self.current_timestamp:
            # 1. Create base order proposals
            proposal = self.create_base_proposal()
            # 2. Apply functions that modify orders price
            self.apply_order_price_modifiers(proposal)
            # 3. Apply functions that modify orders size
            self.apply_order_size_modifiers(proposal)
            # 4. Apply budget constraint, i.e. can't buy/sell more than what you have.
            self.apply_budget_constraint(proposal)

        self.cancel_active_orders_on_max_age_limit()
        self.cancel_active_orders(proposal)

        if self.to_create_orders(proposal):
            self.execute_orders_proposal(proposal)

    @property
    def market(self):
        return self._market_info.market

    @property
    def base_asset(self):
        return self._market_info.base_asset

    @property
    def quote_asset(self):
        return self._market_info.quote_asset

    @property
    def trading_pair(self):
        return self._market_info.trading_pair

    @property
    def market_info_to_active_orders(
        self,
    ) -> Dict[MarketTradingPairTuple, List[LimitOrder]]:
        return self._sb_order_tracker.market_pair_to_active_orders

    @property
    def active_orders(self) -> List[LimitOrder]:
        if self._market_info in self.market_info_to_active_orders:
            return self.market_info_to_active_orders[self._market_info]
        return []

    @property
    def active_buys(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if o.is_buy]

    @property
    def active_sells(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if not o.is_buy]

    def create_base_proposal(self):
        """
        Create base proposal with price at bid_level and ask_level from order book
        """
        buys = []
        sells = []

        # bid_price proposal
        bid_price = Decimal(
            str(
                self._market_info.order_book.snapshot[0].price.iloc[self._bid_level - 1]
            )
        )
        buys.append(PriceSize(bid_price, self._order_amount))

        # ask_price proposal
        ask_price = Decimal(
            str(
                self._market_info.order_book.snapshot[1].price.iloc[self._ask_level - 1]
            )
        )
        sells.append(PriceSize(ask_price, self._order_amount))

        base_proposal = Proposal(buys, sells)
        self.logger().debug(f"Created base proposal: {base_proposal}")
        return base_proposal

    def apply_order_price_modifiers(self, proposal: Proposal):
        self.logger().debug(f"Applied order price modifiers to proposal: {proposal}")

    def apply_order_size_modifiers(self, proposal: Proposal):
        self.logger().debug(f"Applied order size modifiers to proposal: {proposal}")

    def apply_budget_constraint(self, proposal: Proposal):
        """
        Calculate available budget on each asset
        """
        base_balance = self.market.get_available_balance(self.base_asset)
        quote_balance = self.market.get_available_balance(self.quote_asset)

        for buy in proposal.buys:
            buy_fee = self.market.get_fee(
                self.base_asset,
                self.quote_asset,
                OrderType.LIMIT,
                TradeType.BUY,
                buy.size,
                buy.price,
            )
            quote_size = buy.size * buy.price * (Decimal("1") + buy_fee.percent)

            # Adjust buy order size to use remaining balance if less than the order amount
            if quote_balance < quote_size:
                adjusted_amount = quote_balance / (
                    buy.price * (Decimal("1") + buy_fee.percent)
                )
                adjusted_amount = self.market.quantize_order_amount(
                    self.trading_pair, adjusted_amount
                )
                buy.size = adjusted_amount
                quote_balance = s_decimal_zero
            elif quote_balance == s_decimal_zero:
                buy.size = s_decimal_zero
            else:
                quote_balance -= quote_size

        # Filter for valid buys
        proposal.buys = [o for o in proposal.buys if o.size > 0]

        for sell in proposal.sells:
            base_size = sell.size

            # Adjust sell order size to use remaining balance if less than the order amount
            if base_balance < base_size:
                adjusted_amount = self.market.quantize_order_amount(
                    self.trading_pair, base_balance
                )
                sell.size = adjusted_amount
                base_balance = s_decimal_zero
            elif base_balance == s_decimal_zero:
                sell.size = s_decimal_zero
            else:
                base_balance -= base_size

        # Filter for valid sells
        proposal.sells = [o for o in proposal.sells if o.size > 0]

        self.logger().debug(f"Applied budget constraint to proposal: {proposal}")

    def cancel_active_orders_on_max_age_limit(self):
        """
        Cancel active orders if they are older than max age limit
        """
        if (
            self.active_orders
            and self.current_timestamp - self._created_timestamp > self._max_order_age
        ):
            for order in self.active_orders:
                self.cancel_order(self._market_info, order.client_order_id)
            self.logger().info("Cancelled active orders due to max_age_limit")

    def cancel_active_orders(self, proposal: Proposal):
        """
        Cancel active orders, checks if the order prices are at correct levels
        """
        if (
            proposal is not None
            and len(self.active_buys) > 0
            and len(self.active_sells) > 0
        ):
            active_buy_prices = [Decimal(str(o.price)) for o in self.active_buys]
            active_sell_prices = [Decimal(str(o.price)) for o in self.active_sells]
            proposal_buy_prices = [buy.price for buy in proposal.buys]
            proposal_sell_prices = [sell.price for sell in proposal.sells]
            if (
                active_buy_prices != proposal_buy_prices
                or active_sell_prices != proposal_sell_prices
            ):
                for order in self.active_orders:
                    self.cancel_order(self._market_info, order.client_order_id)

    def to_create_orders(self, proposal: Proposal):
        return (
            proposal is not None
            and len(self._sb_order_tracker.in_flight_cancels) == 0
            and (
                len(self._sb_order_tracker.active_asks) == 0
                or len(self._sb_order_tracker.active_bids) == 0
            )
        )

    def execute_orders_proposal(self, proposal: Proposal):
        sell_order_id = None

        if len(proposal.sells) > 0:
            for sell in proposal.sells:
                sell_order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    amount=self._order_amount,
                    order_type=OrderType.LIMIT,
                    price=sell.price,
                )
                self.logger().info(f"Submitted limit sell order {sell_order_id}")

        if sell_order_id is not None and len(proposal.buys) > 0:
            for buy in proposal.buys:
                buy_order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    amount=self._order_amount,
                    order_type=OrderType.LIMIT,
                    price=buy.price,
                )
                self.logger().info(f"Submitted limit buy order {buy_order_id}")

    def did_create_buy_order(self, order_created_event):
        self._created_timestamp = self.current_timestamp

    def did_create_sell_order(self, order_created_event):
        self._created_timestamp = self.current_timestamp

    def did_complete_buy_order(self, order_completed_event):
        self.logger().info(order_completed_event)
        self._create_timestamp = self.current_timestamp + self._filled_order_delay

    def did_complete_sell_order(self, order_completed_event):
        self.logger().info(order_completed_event)
        self._create_timestamp = self.current_timestamp + self._filled_order_delay
