#!/usr/bin/env python

import logging
from datetime import datetime
from decimal import Decimal
from statistics import mean
from typing import Dict, List, Union

import pandas as pd

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    OrderType,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
    TradeType,
)
from hummingbot.core.utils import map_df_to_str
from hummingbot.logger import HummingbotLogger
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
        max_order_age: int,
        filled_order_delay: int,
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
        self._max_order_age = max_order_age
        self._filled_order_delay = filled_order_delay
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
    def has_active_orders(self) -> bool:
        return len(self.active_orders) > 0

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
                level = self.bids_df["price"].index[self.bids_df["price"] <= order.price].to_list()[0] + 1
                side = "buy"
            else:
                level = self.asks_df["price"].index[self.asks_df["price"] >= order.price].to_list()[0] + 1
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
    def in_flight_cancels(self) -> Dict[str, float]:
        return self._sb_order_tracker.in_flight_cancels

    @property
    def order_book(self) -> OrderBook:
        return self._market_info.order_book

    @property
    def asks_df(self) -> pd.DataFrame:
        return self.order_book.snapshot[1]

    @property
    def bids_df(self) -> pd.DataFrame:
        return self.order_book.snapshot[0]

    @property
    def order_amount_in_base_asset(self) -> Decimal:
        return self._order_amount / self.mid_price

    @property
    def order_amount_in_quote_asset(self):
        fee = self.get_fee(self._order_amount)
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
    def is_within_tolerance(self) -> bool:
        return (
            not self._volatility.is_nan()
            and self._volatility <= self._max_volatility / Decimal("100")
        )

    @property
    def is_shield_not_being_activated(self) -> bool:
        return self._create_timestamp < self.current_timestamp

    @property
    def shields(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                ["Time", not self.is_shield_not_being_activated],
                ["Volatility", not self.is_within_tolerance],
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
                    self.logger().warning(
                        f"{self.market.name} is not ready. Please wait..."
                    )
                    return
                else:
                    self.logger().warning(f"{self.market.name} is ready. Trading started.")

            # Cancel orders by max age policy
            self.cancel_active_orders_by_max_order_age()

            # Calculate volatility
            self.update_mid_prices()
            self.update_volatility()

            proposal = None
            if self.is_shield_not_being_activated:
                # Create base order proposals
                proposal = self.create_base_proposal()
                # Cancel active orders based on proposal prices
                self.cancel_active_orders(proposal)
                # Apply budget reallocation
                if self.is_within_tolerance and self.has_active_orders is False:
                    self.apply_budget_reallocation()
                # Apply functions that modify orders price
                # self.apply_order_price_modifiers(proposal)
                # Apply functions that modify orders amount
                # self.apply_order_amount_modifiers(proposal)
                # Apply budget constraint, i.e. can't buy/sell more than what you have.
                self.apply_budget_constraint(proposal)

            if self.to_create_orders(proposal):
                self.execute_orders_proposal(proposal)
        except Exception as exc:
            self.logger().error(f"Unhandled exception in tick function: {exc}")

    def get_fee(self, amount: Decimal):
        """
        Calculate fee based on order amount
        """
        return self.market.get_fee(
            self.base_asset,
            self.quote_asset,
            OrderType.LIMIT,
            TradeType.BUY,
            amount,
            self.mid_price,
        )

    def update_mid_prices(self):
        """
        Query asset markets for mid price
        """
        self._mid_prices.append(self.mid_price)
        # To avoid memory leak, we store only the last part of the list needed for volatility calculation
        max_len = self._volatility_interval * self._avg_volatility_period
        self._mid_prices = self._mid_prices[-1 * max_len:]

    def update_volatility(self):
        """
        Update volatility data from the market
        """
        last_index = len(self._mid_prices) - 1
        atr = []
        first_index = last_index - (
            self._volatility_interval * self._avg_volatility_period
        )
        first_index = max(first_index, 0)
        for i in range(last_index, first_index, self._volatility_interval * -1):
            prices = self._mid_prices[i - self._volatility_interval + 1: i + 1]
            if not prices:
                break
            atr.append((max(prices) - min(prices)) / min(prices))
        if atr:
            self._volatility = mean(atr)
        if (
            self._last_vol_reported
            <= self.current_timestamp - self._volatility_interval
        ):
            if not self._volatility.is_nan():
                self.logger().info(
                    f"{self.trading_pair} volatility: {self._volatility:.2%}"
                )
            self._last_vol_reported = self.current_timestamp

    def create_base_proposal(self) -> Proposal:
        """
        Create base proposal with price at bid_level and ask_level from order book
        """
        buys = []
        sells = []

        # bid_price proposal
        bid_price = Decimal(str(self.bids_df["price"].iloc[self._bid_level - 1]))
        buys.append(PriceSize(bid_price, self.order_amount_in_base_asset))

        # ask_price proposal
        ask_price = Decimal(str(self.asks_df["price"].iloc[self._ask_level - 1]))
        sells.append(PriceSize(ask_price, self.order_amount_in_base_asset))

        base_proposal = Proposal(buys, sells)
        self.logger().debug(f"Created base proposal: {base_proposal}")
        return base_proposal

    def apply_budget_reallocation(self):
        """
        Reallocate quote & base assets to be able to create both BUY and SELL orders
        """
        base_balance = self.market.get_balance(self.base_asset)
        base_balance_in_quote_asset = base_balance * self.best_bid_price
        quote_balance = self.market.get_available_balance(self.quote_asset)
        if (
            base_balance_in_quote_asset
            <= self.order_amount_in_quote_asset - self.min_quote_amount
        ):
            # This allows buying a portion of the base asset
            self.logger().info(
                f"Base asset available balance is low {base_balance} {self.base_asset}."
            )
            amount = max(
                min(
                    self.order_amount_in_base_asset - base_balance, self.best_ask_amount
                ),
                self.min_base_amount,
            )
            order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=self.best_ask_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif (
            base_balance_in_quote_asset + self.order_amount_in_quote_asset
            >= self._budget_allocation
        ):
            # This allows selling a portion of the base asset
            self.logger().info(
                f"Exceeded budget allocation of {self._budget_allocation} {self.quote_asset}."
            )
            amount = max(
                min(
                    base_balance - self.order_amount_in_base_asset, self.best_bid_amount
                ),
                self.min_base_amount,
            )
            order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=amount,
                price=self.best_bid_price,
            )
            self._budget_reallocation_orders.append(order_id)
        elif (
            base_balance >= self.min_base_amount * 2
            and quote_balance < self.min_quote_amount
        ):
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
        elif (
            base_balance < self.min_base_amount
            and quote_balance >= self.min_quote_amount * 2
        ):
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
            self.logger().error(
                "Insufficient balance! Require at least:"
                f"- {self.min_base_amount} {self.base_asset} for SELL side"
                f"- Current: {base_balance} {self.base_asset}"
                f"- {self.min_quote_amount} {self.quote_asset} for BUY side"
                f"- Current: {quote_balance} {self.quote_asset}"
            )
        else:
            # Clear all budget reallocation orders if there is no need to reallocate
            self._budget_reallocation_orders = []

    def apply_order_price_modifiers(self, proposal: Proposal):
        # TODO: implement price modifiers
        self.logger().debug(f"Applied order price modifiers to proposal: {proposal}")

    def apply_order_amount_modifiers(self, proposal: Proposal):
        # TODO: implement amount modifiers
        self.logger().debug(f"Applied order amount modifiers to proposal: {proposal}")

    def apply_budget_constraint(self, proposal: Proposal):
        """
        Calculate available budget on each asset for multiple levels of orders
        """
        base_balance = self.market.get_available_balance(self.base_asset)
        for sell in proposal.sells:
            # Adjust sell order amount to use remaining balance
            # TODO: remove this if you want to implement multi orders
            sell.size = self.market.quantize_order_amount(
                self.trading_pair, base_balance
            )

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
                adjusted_amount = quote_balance / (
                    buy.price * (Decimal("1") + buy_fee.percent)
                )
                adjusted_amount = self.market.quantize_order_amount(
                    self.trading_pair, adjusted_amount
                )
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
        """
        self._created_timestamp = 0  # reset created timestamp
        for order in self.active_orders:
            if order.client_order_id not in self.in_flight_cancels.keys():
                self.cancel_order(self._market_info, order.client_order_id)

    def cancel_active_orders_by_max_order_age(self):
        """
        Cancel active orders if they are older than max age limit
        """
        if (
            self.has_active_orders
            and self._created_timestamp != 0
            and self.current_timestamp - self._created_timestamp > self._max_order_age
        ):
            self._cancel_active_orders()
            self.logger().info("Cancelled active orders due to max_order_age.")

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
                self._cancel_active_orders()

    def to_create_orders(self, proposal: Proposal) -> bool:
        """
        Check all the criteria to create orders
        """
        return (
            proposal is not None
            and len(self.in_flight_cancels) == 0
            and (len(self.active_buys) == 0 or len(self.active_sells) == 0)
            and (len(proposal.buys) > 0 and len(proposal.sells) > 0)
            and self.is_applied_budget_reallocation is False
            and self.is_within_tolerance
        )

    def execute_orders_proposal(self, proposal: Proposal):
        """
        Convert proposal to orders and execute them
        """
        sell_order_id = None
        buy_order_id = None
        if len(proposal.sells) > 0:
            for sell in proposal.sells:
                sell_order_id = self.sell_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    order_type=OrderType.LIMIT,
                    amount=sell.size,
                    price=sell.price,
                )
        if sell_order_id is not None and len(proposal.buys) > 0:
            for buy in proposal.buys:
                buy_order_id = self.buy_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    order_type=OrderType.LIMIT,
                    amount=buy.size,
                    price=buy.price,
                )
        if buy_order_id is not None:
            # If everything is fine, we can clear the budget_reallocation_orders
            self._budget_reallocation_orders = []

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
        fees = ", ".join(
            [
                f"{fee.amount} {fee.token}"
                for fee in order_filled_event.trade_fee.flat_fees
            ]
        )
        messages = [
            f"{order_filled_event.trade_type.name} order filled",
            f"Price: {order_filled_event.price} {self.quote_asset}",
            f"Amount: {order_filled_event.amount} {self.base_asset}",
            f"Fee: {fees}",
        ]
        if order_filled_event.order_id not in self._budget_reallocation_orders:
            self.shield_up()
        else:
            messages.append("Budget reallocation applied, not activating shield.")
        self.notify(messages)
        # Cancel all orders even if they are in the budget_reallocation_orders
        # - prevent partially unfilled orders if applying budget reallocation
        # - prevent filling more asset if not in budget reallocation phase
        self._cancel_active_orders()

    def did_complete_buy_order(self, order_complete_event: BuyOrderCompletedEvent):
        """
        A buy order has been completed.
        """
        messages = [
            "BUY order completed",
            f"{order_complete_event.base_asset_amount} {order_complete_event.base_asset} "
            f"({order_complete_event.quote_asset_amount} {order_complete_event.quote_asset})",
        ]
        if order_complete_event.order_id not in self._budget_reallocation_orders:
            self.shield_up()
        else:
            messages.append("Budget reallocation applied, not activating shield.")
        self.notify(messages)
        self._cancel_active_orders()

    def did_complete_sell_order(self, order_complete_event: SellOrderCompletedEvent):
        """
        A sell order has been completed.
        """
        messages = [
            "SELL order completed",
            f"{order_complete_event.base_asset_amount} {order_complete_event.base_asset} "
            f"({order_complete_event.quote_asset_amount} {order_complete_event.quote_asset})",
        ]
        if order_complete_event.order_id not in self._budget_reallocation_orders:
            self.shield_up()
        else:
            messages.append("Budget reallocation applied, not activating shield.")
        self.notify(messages)
        self._cancel_active_orders()

    def did_cancel_order(self, cancelled_event: OrderCancelledEvent):
        """
        An order has been cancelled. Argument is a OrderCancelledEvent object.
        """
        # Cancel all orders if there is some manually cancelled order
        self._cancel_active_orders()

    def shield_up(self):
        """
        Activate shield unless budget reallocation
        """
        if (
            self.is_applied_budget_reallocation is False
            and self.is_shield_not_being_activated
        ):
            self._create_timestamp = self.current_timestamp + self._filled_order_delay
            until = datetime.fromtimestamp(self._create_timestamp)
            self.notify(f"Shielded up until {until} {until.astimezone().tzname()}.")

    def notify(self, messages: Union[list, str], separator: str = ". "):
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
        lines.extend(
            ["", "Markets:"]
            + ["    " + line for line in markets_df.to_string(index=False).split("\n")]
        )

        # Volatility
        shields_df = map_df_to_str(self.shields)
        lines.extend(
            ["", f"Volatility: {self._volatility:.2%}"]
            + ["    " + line for line in shields_df.to_string(index=False).split("\n")]
        )

        # Current trading balance
        wallet_df = map_df_to_str(self.wallet_balance_data_frame([self._market_info]))
        lines.extend(
            ["", "Balance:"]
            + ["    " + line for line in wallet_df.to_string(index=False).split("\n")]
        )

        # Current active orders
        if self.has_active_orders:
            orders_df = map_df_to_str(self.active_orders_data_frame)
            lines.extend(
                ["", "Orders:"]
                + [
                    "    " + line
                    for line in orders_df.to_string(index=False).split("\n")
                ]
            )
        else:
            lines.extend(["", "No active maker orders."])

        warning_lines.extend(self.balance_warning([self._market_info]))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
