#!/usr/bin/env python

import logging
from datetime import datetime
from decimal import Decimal
from statistics import mean
from typing import List, Optional, Union

import pandas as pd

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderType,
    SellOrderCreatedEvent,
    TradeType,
)
from hummingbot.core.utils import map_df_to_str
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.hunger.types import LevelType
from hummingbot.strategy.hunger.utils import round_non_zero
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.pure_market_making.pure_market_making import PriceSize, Proposal
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.strategy.utils import order_age

DECIMAL_ZERO = Decimal("0")
DECIMAL_NAN = Decimal("NaN")
hws_logger = None


class HungerStrategy(StrategyPyBase):
    # We use StrategyPyBase to inherit the structure. We also
    # create a logger object before adding a constructor to the class.
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exchange_ready = False
        # Config map
        self._filled_order_delay = None
        self._max_order_age = None
        self._bid_level = None
        self._ask_level = None
        self._budget_allocation = None
        self._order_amount = None
        self._market_info = None
        self._volatility_interval = None
        self._avg_volatility_period = None
        self._max_volatility = None
        # Global timestamp
        self._create_timestamp = 0
        self._created_timestamp = 0
        self._cancelled_timestamp = 0
        # Global states
        self._budget_reallocation_orders = []
        self._mid_prices = []
        self._volatility = DECIMAL_NAN
        self._last_vol_reported = 0

    @classmethod
    def logger(cls) -> HummingbotLogger:
        global hws_logger
        if hws_logger is None:
            hws_logger = logging.getLogger(__name__)
        return hws_logger

    def init_params(
        self,
        market_info: MarketTradingPairTuple,
        order_amount: Decimal,
        budget_allocation: Decimal,
        ask_level: int,
        bid_level: int,
        realtime_levels_enabled: bool = False,
        max_order_age: int = 600,
        filled_order_delay: int = 300,
        order_refresh_tolerance_pct: Decimal = Decimal("0.2"),
        volatility_interval: int = 60,
        avg_volatility_period: int = 5,
        max_volatility: Decimal = DECIMAL_ZERO,
    ):
        super().__init__()
        self._market_info = market_info
        self._order_amount = order_amount
        self._budget_allocation = budget_allocation
        self._ask_level = ask_level
        self._bid_level = bid_level
        self._realtime_levels_enabled = realtime_levels_enabled
        self._max_order_age = max_order_age
        self._filled_order_delay = filled_order_delay
        self._order_refresh_tolerance_pct = order_refresh_tolerance_pct
        self._volatility_interval = volatility_interval
        self._avg_volatility_period = avg_volatility_period
        self._max_volatility = max_volatility
        self.add_markets([market_info.market])

    @property
    def market(self) -> ExchangeBase:
        return self._market_info.market

    @property
    def base_asset(self) -> str:
        return self._market_info.base_asset

    @property
    def quote_asset(self) -> str:
        return self._market_info.quote_asset

    @property
    def trading_pair(self) -> str:
        return self._market_info.trading_pair

    @property
    def active_orders(self) -> List[LimitOrder]:
        return [o[1] for o in self.order_tracker.active_limit_orders]

    @property
    def has_active_orders(self) -> bool:
        return len(self.active_orders) > 0

    @property
    def has_in_flight_cancels(self) -> bool:
        return len(self.order_tracker.in_flight_cancels) > 0

    @property
    def active_orders_data_frame(self) -> pd.DataFrame:
        active_orders = self.active_orders
        active_orders.sort(key=lambda x: x.price, reverse=True)
        data = []
        for order in active_orders:
            # Calculate actual spread
            spread = abs(order.price - self.mid_price) / self.mid_price
            # Calculate current order age
            age = pd.Timestamp(order_age(order, self.current_timestamp), unit="s").strftime("%H:%M:%S")
            # Find actual order levels on orderbook
            if order.is_buy:
                series = self.bids_df["price"].index[self.bids_df["price"] == float(order.price)]
                level = series.to_list()[0] + 1
                side = "buy"
            else:
                series = self.asks_df["price"].index[self.asks_df["price"] == float(order.price)]
                level = series.to_list()[0] + 1
                side = "sell"
            data.append(
                [
                    level,
                    side,
                    float(order.price),
                    f"{spread:.2%}",
                    float(order.quantity),
                    age,
                ]
            )
        columns = ["Level", "Type", "Price", "Spread", "Amount", "Age"]
        return pd.DataFrame(columns=columns, data=data)

    @property
    def active_buys(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if o.is_buy]

    @property
    def active_sells(self) -> List[LimitOrder]:
        return [o for o in self.active_orders if not o.is_buy]

    @property
    def asks_df(self) -> pd.DataFrame:
        return self._market_info.order_book.snapshot[1]

    @property
    def bids_df(self) -> pd.DataFrame:
        return self._market_info.order_book.snapshot[0]

    @property
    def ask_level_type(self) -> LevelType:
        return LevelType.from_str(self._ask_level)

    @property
    def ask_level(self) -> Decimal:
        return LevelType.to_decimal(self._ask_level)

    @property
    def bid_level_type(self) -> LevelType:
        return LevelType.from_str(self._bid_level)

    @property
    def bid_level(self) -> Decimal:
        return LevelType.to_decimal(self._bid_level)

    @property
    def order_amount_in_base_asset(self) -> Decimal:
        return self._order_amount / self.mid_price

    @property
    def order_amount_in_quote_asset(self):
        fee = self.market.get_fee(
            self.base_asset,
            self.quote_asset,
            OrderType.LIMIT,
            TradeType.BUY,
            self._order_amount,
            self.mid_price,
        )
        return self._order_amount * (Decimal("1") + fee.percent)

    @property
    def best_ask_price(self) -> Decimal:
        return Decimal(str(self.asks_df["price"].iloc[0]))

    @property
    def best_ask_amount(self) -> Decimal:
        return Decimal(str(self.asks_df["amount"].iloc[0]))

    @property
    def mid_price(self) -> Decimal:
        return self._market_info.get_mid_price()

    @property
    def best_bid_price(self) -> Decimal:
        return Decimal(str(self.bids_df["price"].iloc[0]))

    @property
    def best_bid_amount(self) -> Decimal:
        return Decimal(str(self.bids_df["amount"].iloc[0]))

    @property
    def trading_rule(self) -> TradingRule:
        return self.market.trading_rules[self.trading_pair]

    @property
    def min_base_amount(self) -> Decimal:
        return self.min_quote_amount / self.mid_price

    @property
    def min_quote_amount(self) -> Decimal:
        if self.trading_rule.min_order_size > DECIMAL_ZERO:
            # For pairs of coin-coin, ex: VSP-ETH, HMT-ETH
            amount = self.trading_rule.min_order_size
        else:
            # For pairs of coin-stable, ex: AVAX-USDT
            amount = self.trading_rule.min_notional_size
        # Min quote amount must not be less than $5
        return max(amount * Decimal("1.1"), Decimal("5"))

    @property
    def is_not_volatile(self) -> bool:
        return (
            not self._volatility.is_nan()
            and type(self._max_volatility) is Decimal
            and self._volatility <= self._max_volatility / Decimal("100")
        )

    @property
    def is_volatile(self) -> bool:
        return not self.is_not_volatile

    @property
    def is_time_shield_not_being_activated(self) -> bool:
        return self._create_timestamp < self.current_timestamp

    @property
    def is_time_shield_being_activated(self) -> bool:
        return not self.is_time_shield_not_being_activated

    @property
    def shields(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["Time", self.is_time_shield_being_activated],
                ["Volatility", self.is_volatile],
            ],
            columns=[
                "Shield",
                "Status",
            ],
        )

    @property
    def is_applied_budget_reallocation(self) -> bool:
        return len(self._budget_reallocation_orders) > 0

    # After initializing the required variables, we define the tick method.
    # The tick method is the entry point for the strategy.
    def tick(self, timestamp: float):
        try:
            if not self._exchange_ready:
                self._exchange_ready = self.market.ready
                if not self._exchange_ready:
                    self.logger().warning(f"{self.market.name} is not ready. Please wait...")
                    return
                else:
                    self.logger().warning(f"{self.market.name} is ready. Trading started.")

            # Cancel orders by max age policy
            self.cancel_active_orders_by_max_order_age()

            # Calculate volatility
            self.update_mid_prices()
            self.update_volatility()

            if self.is_time_shield_not_being_activated:
                # Create base order proposals
                proposal = self.create_base_proposal()
                # Cancel active orders based on proposal prices
                self.cancel_active_orders(proposal)
                # Apply budget reallocation
                self.apply_budget_reallocation()
                # Apply budget constraint, i.e. can't buy/sell more than what you have.
                self.apply_budget_constraint(proposal)

                # Whether to create new orders or not
                if self.to_create_orders(proposal):
                    self.execute_orders_proposal(proposal)
        except Exception as exc:
            self.logger().error(f"Unhandled exception in tick function: {str(exc)}", exc_info=True)

    def update_mid_prices(self):
        """
        Query asset markets for mid price
        """
        self._mid_prices.append(self.mid_price)
        # To avoid memory leak, we store only the last part of the list needed for volatility calculation
        if type(self._volatility_interval) is int and type(self._avg_volatility_period) is int:
            max_len = self._volatility_interval * self._avg_volatility_period
            self._mid_prices = self._mid_prices[-1 * max_len:]

    def update_volatility(self):
        """
        Update volatility data from the market
        """
        if type(self._volatility_interval) is int and type(self._avg_volatility_period) is int:
            last_index = len(self._mid_prices) - 1
            atr = []
            first_index = last_index - (self._volatility_interval * self._avg_volatility_period)
            first_index = max(first_index, 0)
            for i in range(last_index, first_index, self._volatility_interval * -1):
                prices = self._mid_prices[i - self._volatility_interval + 1: i + 1]
                if not prices:
                    break
                atr.append((max(prices) - min(prices)) / min(prices))
            if atr:
                self._volatility = mean(atr)
            if self._last_vol_reported <= self.current_timestamp - self._volatility_interval:
                if not self._volatility.is_nan():
                    self.logger().info(f"{self.trading_pair} volatility: {self._volatility:.2%}")
                self._last_vol_reported = self.current_timestamp

    def create_base_bid_price(self) -> Optional[Decimal]:
        """
        Create base bid price
        """
        if self.bid_level_type is LevelType.INTEGER:
            return Decimal(str(self.bids_df["price"].iloc[self.bid_level - 1]))
        elif self.bid_level_type is LevelType.PERCENTAGE:
            bid_price = self.mid_price * abs(Decimal("1") - self.bid_level / 100)
            return self.market.quantize_order_price(self.trading_pair, bid_price)

    def create_base_ask_price(self) -> Optional[Decimal]:
        """
        Create base ask price
        """
        if self.ask_level_type is LevelType.INTEGER:
            return Decimal(str(self.asks_df["price"].iloc[self.ask_level - 1]))
        elif self.ask_level_type is LevelType.PERCENTAGE:
            ask_price = self.mid_price * abs(Decimal("1") + self.ask_level / 100)
            return self.market.quantize_order_price(self.trading_pair, ask_price)

    def create_base_proposal(self) -> Proposal:
        """
        Create base proposal with price at bid_level and ask_level from order book
        """
        base_proposal = Proposal(
            [PriceSize(self.create_base_bid_price(), self.order_amount_in_base_asset)],  # bids
            [PriceSize(self.create_base_ask_price(), self.order_amount_in_base_asset)],  # asks
        )
        self.logger().debug(f"Created base proposal: {base_proposal}")
        return base_proposal

    def apply_budget_reallocation(self):
        """
        Reallocate quote & base assets to be able to create both BUY and SELL orders
        """
        if self.is_not_volatile and self.has_active_orders is False:
            return
        base_balance = self.market.get_balance(self.base_asset)
        base_balance_in_quote_asset = base_balance * self.best_bid_price
        quote_balance = self.market.get_available_balance(self.quote_asset)
        if base_balance_in_quote_asset <= self.order_amount_in_quote_asset - self.min_quote_amount:
            # This allows buying a portion of the base asset
            self.logger().info(f"Base asset available balance is low {base_balance} {self.base_asset}.")
            amount = max(
                min(self.order_amount_in_base_asset - base_balance, self.best_ask_amount),
                self.min_base_amount,
            )
            order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=self.best_ask_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif base_balance_in_quote_asset + self.order_amount_in_quote_asset >= self._budget_allocation:
            # This allows selling a portion of the base asset
            self.logger().info(f"Exceeded budget allocation of {self._budget_allocation} {self.quote_asset}.")
            amount = max(
                min(base_balance - self.order_amount_in_base_asset, self.best_bid_amount),
                self.min_base_amount,
            )
            order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=self.best_bid_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif base_balance >= self.min_base_amount * 2 and quote_balance < self.min_quote_amount:
            # This allows selling a portion of the base asset
            messages = [
                f"Quote asset balance is too low - {quote_balance} {self.quote_asset}",
                f"Minimum require: {self.min_quote_amount} {self.quote_asset}",
                f"Order amount: {self.order_amount_in_quote_asset} {self.quote_asset}",
            ]
            self.logger().info(". ".join(messages))
            order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self.min_base_amount,
                price=self.best_bid_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif base_balance < self.min_base_amount and quote_balance >= self.min_quote_amount * 2:
            # This allows buying a portion of the base asset
            messages = [
                f"Base asset balance is too low - {base_balance} {self.base_asset}",
                f"Minimum require: {self.min_base_amount} {self.base_asset}",
                f"Order amount: {self.order_amount_in_base_asset} {self.base_asset}",
            ]
            self.logger().info(". ".join(messages))
            # If there is pretty low quote asset
            order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self.min_base_amount,
                price=self.best_ask_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif base_balance_in_quote_asset + quote_balance < self.min_quote_amount * 2:
            # Balance is too low to create both BUY and SELL orders
            self.logger().info(
                "Insufficient balance! Require at least:"
                f"- {self.min_base_amount} {self.base_asset} for SELL side"
                f"- Current: {base_balance} {self.base_asset}"
                f"- {self.min_quote_amount} {self.quote_asset} for BUY side"
                f"- Current: {quote_balance} {self.quote_asset}"
            )

    def apply_budget_constraint(self, proposal: Proposal):
        """
        Calculate available budget on each asset for multiple levels of orders
        """
        # FIXME: gate_io has problem with get_available_balance
        if self.market.name == "gate_io":
            base_balance = self.market.get_balance(self.base_asset)
        else:
            base_balance = self.market.get_available_balance(self.base_asset)
        for sell in proposal.sells:
            # Adjust sell order amount to use remaining balance
            # FIXME: remove this if you want to implement multi orders
            sell.size = self.market.quantize_order_amount(self.trading_pair, base_balance)

            # Adjust sell order amount to use remaining balance if less than the order amount
            # if base_balance < base_amount:
            #     sell.size = self.market.quantize_order_amount(
            #         self.trading_pair, base_balance
            #     )
            #     base_balance = DECIMAL_ZERO
            # elif base_balance == DECIMAL_ZERO:
            #     sell.size = DECIMAL_ZERO
            # else:
            #     base_balance -= base_amount

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
            quote_amount = buy.size * buy.price * (Decimal("1") + buy_fee.percent)

            # Adjust buy order amount to use remaining balance if less than the order amount
            if quote_balance < quote_amount:
                adjusted_amount = quote_balance / (buy.price * (Decimal("1") + buy_fee.percent))
                adjusted_amount = self.market.quantize_order_amount(self.trading_pair, adjusted_amount)
                buy.size = adjusted_amount
                quote_balance = DECIMAL_ZERO
            elif quote_balance == DECIMAL_ZERO:
                buy.size = DECIMAL_ZERO
            else:
                quote_balance -= quote_amount

        # Filter for valid proposals
        proposal.sells = [o for o in proposal.sells if o.size > 0]
        proposal.buys = [o for o in proposal.buys if o.size > 0]

        self.logger().debug(f"Applied budget constraint to proposal: {proposal}")

    def _cancel_active_orders(self):
        """
        Cancel all active orders if any
        - Cancelling an order might be failed by any reason
        """
        self._created_timestamp = 0  # reset created timestamp
        self._cancelled_timestamp = self.current_timestamp  # last cancelled timestamp

        if self.has_active_orders is True:
            # Cancel all active orders
            for order in self.active_orders:
                self._force_cancel_order(order.client_order_id)
            self.logger().info("Cancelled all active orders.")
        elif (
            self.market.name == "ascend_ex"
            and self.has_in_flight_cancels is False
            and len(self.market.in_flight_orders) > 0
        ):
            # Clear all active orders (ascend_ex only)
            for client_order_id, _ in self.market.in_flight_orders.items():
                self._force_cancel_order(client_order_id)
            self.logger().info("Cancelled in flight orders if any.")

    def _force_cancel_order(self, order_id: str):
        """
        Force cancel an order
        """
        self.cancel_order(self._market_info, order_id)

        #  If no order is found on remote exchange, manually remove it from tracked orders (ascend_ex only)
        if self.market.name == "ascend_ex":
            non_tracked_order = self.market._in_flight_order_tracker.fetch_cached_order(order_id)
            if non_tracked_order is None:
                self.stop_tracking_limit_order(self._market_info, order_id)
                self.logger().info(f"Order {order_id} is not found. Stop tracking.")

    def _is_within_tolerance(self, proposal: Proposal) -> bool:
        """
        False if there are no buys or sells or if the difference between the proposed price and current price is less
        than the tolerance. The tolerance value is strict max, cannot be equal.
        """
        first_buy = proposal.buys[0] if len(proposal.buys) > 0 else None
        first_sell = proposal.sells[0] if len(proposal.sells) > 0 else None
        if (self.active_buys and first_buy.size <= 0) or (self.active_sells and first_sell.size <= 0):
            return False
        if (
            self.active_buys
            and abs(first_buy.price - self.active_buys[0].price) / self.active_buys[0].price
            > self._order_refresh_tolerance_pct
        ):
            return False
        if (
            self.active_sells
            and abs(first_sell.price - self.active_sells[0].price) / self.active_sells[0].price
            > self._order_refresh_tolerance_pct
        ):
            return False
        return True

    def _is_within_correct_levels(self, proposal: Proposal) -> bool:
        if self._realtime_levels_enabled is True and proposal is not None:
            if self.ask_level_type is LevelType.INTEGER:
                for index, sell in enumerate(self.active_sells):
                    # prevent index out of range
                    if index < len(proposal.sells) and sell.price != proposal.sells[index].price:
                        self.logger().info(
                            "Sell price at level {} changed {} {}.".format(
                                index, proposal.sells[index].price, self.quote_asset
                            )
                        )
                        self.logger().info("Cancelled active orders due to realtime_levels_enabled.")
                        return False

            if self.bid_level_type is LevelType.INTEGER:
                for index, buy in enumerate(self.active_buys):
                    # prevent index out of range
                    if index < len(proposal.buys) and buy.price != proposal.buys[index].price:
                        self.logger().info(
                            "Buy price at level {} changed {} {}.".format(
                                index, proposal.sells[index].price, self.quote_asset
                            )
                        )
                        self.logger().info("Cancelled active orders due to realtime_levels_enabled.")
                        return False
        return True

    def cancel_active_orders_by_max_order_age(self):
        """
        Cancel active orders if they are older than max age limit
        """
        if (
            self.has_active_orders is True
            and self._created_timestamp > 0
            and self.current_timestamp - self._created_timestamp > self._max_order_age
        ):
            self._cancel_active_orders()
            self.logger().info("Cancelled active orders due to max_order_age.")

    def cancel_active_orders(self, proposal: Proposal):
        """
        Cancel active orders, checks if the order prices are at correct levels
        """
        should_cancel = False

        # Cancel all active orders by conditions
        if self.has_active_orders is True:
            # Ensure proposal buys and sells are the same as active buys and sells
            if len(proposal.buys) != len(self.active_buys) or len(proposal.sells) != len(self.active_sells):
                should_cancel = True
                self.logger().info("Cancelled active orders due to proposal updates.")

            # Ensure proposal is within tolerance
            if should_cancel is False and self._is_within_tolerance(proposal) is False:
                should_cancel = True
                self.logger().info("Cancelled active orders due to tolerance.")

            # Ensure correct buy/sell levels
            if should_cancel is False and self._is_within_correct_levels(proposal) is False:
                should_cancel = True
                self.logger().info("Cancelled active orders due to incorrect levels.")

            # Ensure number of buys and sells are the same
            if (
                should_cancel is False
                and len(self.active_buys) != len(self.active_sells)
                and self.current_timestamp - self._created_timestamp > 3
            ):
                should_cancel = True
                self.logger().info("Number of buys and sells need to equalize.")

        elif self.has_in_flight_cancels is True and self.current_timestamp - self._cancelled_timestamp > 3:
            # Cancel all in-flight cancels if any
            should_cancel = True
            self.logger().info("Bot get stuck by in_flight_cancels.")

        if should_cancel:
            self._cancel_active_orders()

    def to_create_orders(self, proposal: Proposal) -> bool:
        """
        Check all the criteria to create orders
        """
        return (
            proposal is not None
            and (len(proposal.buys) > 0 and len(proposal.sells) > 0)
            and self._created_timestamp == 0
            and self.has_in_flight_cancels is False
            and self.has_active_orders is False
            and self.is_applied_budget_reallocation is False
            and self.is_volatile is False
        )

    def execute_orders_proposal(self, proposal: Proposal):
        """
        Convert proposal to orders and execute them
        """
        sell_order_ids = []
        buy_order_ids = []
        if len(proposal.sells) > 0:
            for sell in proposal.sells:
                sell_order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    order_type=OrderType.LIMIT,
                    amount=sell.size,
                    price=sell.price,
                )
                sell_order_ids.append(sell_order_id)
            self.logger().info("Created {} sell orders.".format(len(sell_order_ids)))
        if len(sell_order_ids) > 0 and len(proposal.buys) > 0:
            for buy in proposal.buys:
                buy_order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    order_type=OrderType.LIMIT,
                    amount=buy.size,
                    price=buy.price,
                )
                buy_order_ids.append(buy_order_id)
            self.logger().info("Created {} buy orders.".format(len(buy_order_ids)))

    def did_create_buy_order(self, order_created_event: BuyOrderCreatedEvent):
        """
        A buy order has been created. Argument is a BuyOrderCreatedEvent object.
        """
        self._created_timestamp = self.current_timestamp

    def did_create_sell_order(self, order_created_event: SellOrderCreatedEvent):
        """
        A sell order has been created. Argument is a SellOrderCreatedEvent object.
        """
        self._created_timestamp = self.current_timestamp

    def did_fill_order(self, order_filled_event: OrderFilledEvent):
        """
        An order has been filled in the market. Argument is a OrderFilledEvent object.
        """
        messages = [
            f"{order_filled_event.trade_type.name} order filled",
            f"Price: {order_filled_event.price} {self.quote_asset}",
            f"Amount: {order_filled_event.amount} {self.base_asset}",
        ]
        fees = ", ".join([f"{fee.amount} {fee.token}" for fee in order_filled_event.trade_fee.flat_fees])
        if fees:
            messages.append(f"Fees: {fees}")
        if order_filled_event.order_id in self._budget_reallocation_orders:
            # Turn on short shield if the order is a budget reallocation order
            messages.append("Budget reallocation applied, turn short shield up.")
            self._shield_up(5)
            self._budget_reallocation_orders.remove(order_filled_event.order_id)
        else:
            # Turn on default shield if the order is not a budget reallocation order
            self._shield_up()
        self._notify(messages)
        # Cancel all orders even if they are in the budget_reallocation_orders
        # - prevent partially unfilled orders if applying budget reallocation
        # - prevent filling more asset if not in budget reallocation phase
        self._cancel_active_orders()

    # def did_complete_buy_order(self, order_complete_event: BuyOrderCompletedEvent):
    #     """
    #     A buy order has been completed.
    #     """
    #     messages = [
    #         "BUY order completed",
    #         f"{order_complete_event.base_asset_amount} {order_complete_event.base_asset} "
    #         f"({order_complete_event.quote_asset_amount} {order_complete_event.quote_asset})",
    #     ]
    #     if order_complete_event.order_id in self._budget_reallocation_orders:
    #         # Turn on short shield if the order is a budget reallocation order
    #         messages.append("Budget reallocation applied, turn short shield up.")
    #         self._shield_up(5)
    #         self._budget_reallocation_orders.remove(order_complete_event.order_id)
    #     else:
    #         # Turn on default shield if the order is not a budget reallocation order
    #         self._shield_up()
    #     self._notify(messages)
    #     self._cancel_active_orders()

    # def did_complete_sell_order(self, order_complete_event: SellOrderCompletedEvent):
    #     """
    #     A sell order has been completed.
    #     """
    #     messages = [
    #         "SELL order completed",
    #         f"{order_complete_event.base_asset_amount} {order_complete_event.base_asset} "
    #         f"({order_complete_event.quote_asset_amount} {order_complete_event.quote_asset})",
    #     ]
    #     if order_complete_event.order_id in self._budget_reallocation_orders:
    #         # Turn on short shield if the order is a budget reallocation order
    #         messages.append("Budget reallocation applied, turn short shield up.")
    #         self._shield_up(5)
    #         self._budget_reallocation_orders.remove(order_complete_event.order_id)
    #     else:
    #         # Turn on default shield if the order is not a budget reallocation order
    #         self._shield_up()
    #     self._notify(messages)
    #     self._cancel_active_orders()

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        """
        An order has been cancelled. Argument is a OrderCancelledEvent object.
        """
        # Cancel all orders if there is some manually cancelled order
        self._cancel_active_orders()

    def _shield_up(self, delay: int = 0):
        """
        Activate shield unless budget reallocation
        """
        if self.is_time_shield_not_being_activated:
            if type(delay) is int and delay > 0:
                self._create_timestamp = self.current_timestamp + delay
            else:
                self._create_timestamp = self.current_timestamp + self._filled_order_delay
            until = datetime.fromtimestamp(self._create_timestamp)
            self._notify(f"Shielded up until {until} {until.astimezone().tzname()}.")

    def _notify(self, messages: Union[list, str], separator: str = ". "):
        """
        Notify the user via both logger and hb app
        """
        if type(messages) is list:
            messages = separator.join(messages)
        self.logger().info(messages)
        self.notify_hb_app(messages)

    def format_status(self):
        """
        Return the budget, market, miner and order states.
        """
        if not self._exchange_ready:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning([self._market_info]))

        # Current market data
        markets_df = map_df_to_str(self.market_status_data_frame([self._market_info]))
        lines.extend(["", "Markets:"] + ["    " + line for line in markets_df.to_string(index=False).split("\n")])

        # Volatility
        shields_df = map_df_to_str(self.shields)
        lines.extend(
            ["", f"Volatility: {self._volatility:.2%}"]
            + ["    " + line for line in shields_df.to_string(index=False).split("\n")]
        )

        # Current trading balance
        wallet_df = map_df_to_str(self.wallet_balance_data_frame([self._market_info]))
        wallet_df["Total Balance"] = wallet_df["Total Balance"].apply(lambda x: str(round_non_zero(float(x))))
        wallet_df["Available Balance"] = wallet_df["Available Balance"].apply(lambda x: str(round_non_zero(float(x))))
        lines.extend(["", "Balance:"] + ["    " + line for line in wallet_df.to_string(index=False).split("\n")])

        # Current active orders
        if self.has_active_orders is True:
            orders_df = map_df_to_str(self.active_orders_data_frame)
            orders_df["Amount"] = orders_df["Amount"].apply(lambda x: str(round_non_zero(float(x))))
            lines.extend(["", "Orders:"] + ["    " + line for line in orders_df.to_string(index=False).split("\n")])
        else:
            lines.extend(["", "No active maker orders."])

        warning_lines.extend(self.balance_warning([self._market_info]))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
