#!/usr/bin/env python

import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List

import pandas as pd

from hummingbot.connector.exchange_base import ExchangeBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.order_book import OrderBook
from hummingbot.core.event.events import OrderType, TradeType
from hummingbot.core.utils import map_df_to_str
from hummingbot.logger import HummingbotLogger
from hummingbot.strategy.market_trading_pair_tuple import MarketTradingPairTuple
from hummingbot.strategy.pure_market_making.pure_market_making import PriceSize, Proposal
from hummingbot.strategy.strategy_py_base import StrategyPyBase
from hummingbot.strategy.utils import order_age

DECIMAL_ZERO = Decimal("0")
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
        # Global timestamp
        self._create_timestamp = 0
        self._created_timestamp = 0
        # Global states
        self._applied_budget_reallocation = False

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
    ):
        super().__init__()
        self._market_info = market_info
        self._order_amount = order_amount
        self._budget_allocation = budget_allocation
        self._ask_level = ask_level
        self._bid_level = bid_level
        self._max_order_age = max_order_age
        self._filled_order_delay = filled_order_delay
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
    def active_orders_data_frame(self) -> pd.DataFrame:
        active_orders = self.active_orders
        active_orders.sort(key=lambda x: x.price, reverse=True)
        data = []
        for order in active_orders:
            spread = abs(order.price - self.mid_price) / self.mid_price
            age = pd.Timestamp(
                order_age(order, self.current_timestamp), unit="s"
            ).strftime("%H:%M:%S")
            data.append(
                [
                    self._bid_level,
                    "buy" if order.is_buy else "sell",
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
    def order_amount_in_quote_asset(self):
        notional_amount = self._order_amount * self.mid_price
        buy_fee = self.get_fee(notional_amount)
        return notional_amount * (Decimal("1") + buy_fee.percent)

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
    def min_base_amount(self) -> Decimal:
        return self.min_quote_amount / self.mid_price

    @property
    def trading_rule(self) -> TradingRule:
        return self.market.trading_rules[self.trading_pair]

    @property
    def min_quote_amount(self) -> Decimal:
        if self.trading_rule.min_order_size > DECIMAL_ZERO:
            # For pairs of coin-coin, ex: VSP-ETH, HMT-ETH
            amount = self.trading_rule.min_order_size
        else:
            # For paris of coin-stable, ex: AVAX-USDT
            amount = self.trading_rule.min_notional_size
        return amount * Decimal("1.5")

    # After initializing the required variables, we define the tick method.
    # The tick method is the entry point for the strategy.
    def tick(self, timestamp: float):
        if not self._exchange_ready:
            self._exchange_ready = self.market.ready
            if not self._exchange_ready:
                self.logger().warning(
                    f"{self.market.name} is not ready. Please wait..."
                )
                return
            else:
                self.logger().warning(f"{self.market.name} is ready. Trading started")

        # Cancel orders by max age policy
        self.cancel_active_orders_by_max_order_age()

        proposal = None
        if self._create_timestamp < self.current_timestamp:
            # Create base order proposals
            proposal = self.create_base_proposal()
            # Cancel active orders based on proposal prices
            self.cancel_active_orders(proposal)
            # Apply budget reallocation
            self.apply_budget_reallocation()
            # Apply functions that modify orders price
            # self.apply_order_price_modifiers(proposal)
            # Apply functions that modify orders amount
            # self.apply_order_amount_modifiers(proposal)
            # Apply budget constraint, i.e. can't buy/sell more than what you have.
            self.apply_budget_constraint(proposal)

        if self.to_create_orders(proposal):
            self.execute_orders_proposal(proposal)

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

    def create_base_proposal(self) -> Proposal:
        """
        Create base proposal with price at bid_level and ask_level from order book
        """
        buys = []
        sells = []

        # bid_price proposal
        bid_price = Decimal(str(self.bids_df["price"].iloc[self._bid_level - 1]))
        buys.append(PriceSize(bid_price, self._order_amount))

        # ask_price proposal
        ask_price = Decimal(str(self.asks_df["price"].iloc[self._ask_level - 1]))
        sells.append(PriceSize(ask_price, self._order_amount))

        base_proposal = Proposal(buys, sells)
        self.logger().debug(f"Created base proposal: {base_proposal}")
        return base_proposal

    def apply_budget_reallocation(self):
        """
        Reallocate quote & base assets to be able to create both BUY and SELL orders
        """
        base_balance = self._market_info.base_balance
        base_balance_in_quote_asset = base_balance * self.best_bid_price
        quote_balance = self._market_info.quote_balance
        if (
            base_balance_in_quote_asset + self.order_amount_in_quote_asset
            >= self._budget_allocation
        ):
            # This allows selling a portion of the base asset
            self.logger().info(
                f"Exceeded budget allocation of {self._budget_allocation} {self.quote_asset}"
            )
            self.handle_insufficient_quote_balance_error()
        elif (
            quote_balance < self.min_quote_amount
            and base_balance >= self.min_base_amount * 2
        ):
            # This allows selling a portion of the base asset
            self.logger().info(
                f"Quote asset balance is too low - {quote_balance} {self.quote_asset}\n"
                f"- Minimum require: {self.min_quote_amount} {self.quote_asset}\n"
                f"- Order amount: {self.order_amount_in_quote_asset} {self.quote_asset}"
            )
            self.handle_insufficient_quote_balance_error()
        elif (
            base_balance < self.min_base_amount
            and quote_balance >= self.min_quote_amount * 2
        ):
            # This allows buying a portion of the base asset
            self.logger().info(
                f"Base asset balance is too low - {base_balance} {self.base_asset}\n"
                f"- Minimum require: {self.min_base_amount} {self.base_asset}\n"
                f"- Order amount: {self._order_amount} {self.base_asset}"
            )
            self.handle_insufficient_base_balance_error()
        elif base_balance_in_quote_asset + quote_balance < self.min_quote_amount * 2:
            self.logger().info(
                "Insufficient balance! Require at least:"
                f"- {self.min_base_amount} {self.base_asset} for SELL side"
                f"- Current: {base_balance} {self.base_asset}"
                f"- {self.min_quote_amount} {self.quote_asset} for BUY side"
                f"- Current: {quote_balance} {self.quote_asset}"
            )

    def handle_insufficient_base_balance_error(self):
        """
        Re-balance assets: market buy a portion of base asset
        """
        # TODO: add volatility calculation before placing a buy order
        base_balance = self.market.get_available_balance(self.base_asset)
        quote_balance = self.market.get_available_balance(self.quote_asset)
        buy_order_id = None
        if quote_balance >= self.order_amount_in_quote_asset * 2:
            # If there is good amount of quote asset
            # "self._order_amount - base_balance" is to prevent buy too much of base asset
            # if there is still some base assets (but not enough to create SELL order)
            buy_order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self._order_amount - base_balance,
                price=self.best_ask_price,
            )
        elif quote_balance >= self.min_quote_amount * 2:
            # If there is pretty low of quote asset
            buy_order_id = self.buy_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self.min_base_amount,
                price=self.best_ask_price,
            )
        # Update state only if buy successfully
        if buy_order_id is not None:
            self._applied_budget_reallocation = True

    def handle_insufficient_quote_balance_error(self):
        """
        Re-balance assets: market sell a portion of base asset
        """
        base_balance = self.market.get_available_balance(self.base_asset)
        sell_order_id = None
        if base_balance >= self._order_amount * 2:
            sell_order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self._order_amount,
                price=self.best_bid_price,
            )
        elif base_balance >= self.min_base_amount * 2:
            sell_order_id = self.sell_with_specific_market(
                market_trading_pair_tuple=self._market_info,
                order_type=OrderType.LIMIT,
                amount=self.min_base_amount,
                price=self.best_bid_price,
            )
        # Update state only if sell successfully
        if sell_order_id is not None:
            self._applied_budget_reallocation = True

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
            base_amount = sell.size

            # Adjust sell order amount to use remaining balance if less than the order amount
            if base_balance < base_amount:
                adjusted_amount = self.market.quantize_order_amount(
                    self.trading_pair, base_balance
                )
                sell.size = adjusted_amount
                base_balance = DECIMAL_ZERO
            elif base_balance == DECIMAL_ZERO:
                sell.size = DECIMAL_ZERO
            else:
                base_balance -= base_amount

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
        self._created_timestamp = 0
        for order in self.active_orders:
            if order.client_order_id not in self.in_flight_cancels.keys():
                self.cancel_order(self._market_info, order.client_order_id)

    def cancel_active_orders_by_max_order_age(self):
        """
        Cancel active orders if they are older than max age limit
        """
        if (
            self.active_orders
            and self._created_timestamp != 0
            and self.current_timestamp - self._created_timestamp > self._max_order_age
        ):
            self._cancel_active_orders()
            self.logger().info("Cancelled active orders due to max_order_age")

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

    def to_create_orders(self, proposal: Proposal):
        return (
            proposal is not None
            and len(self.in_flight_cancels) == 0
            and (len(self.active_buys) == 0 or len(self.active_sells) == 0)
            and (len(proposal.buys) > 0 and len(proposal.sells) > 0)
        )

    def execute_orders_proposal(self, proposal: Proposal):
        sell_order_id = None

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
                self.buy_with_specific_market(
                    market_trading_pair_tuple=self._market_info,
                    order_type=OrderType.LIMIT,
                    amount=buy.size,
                    price=buy.price,
                )

    def did_create_buy_order(self, order_created_event):
        """
        A buy order has been created. Argument is a BuyOrderCreatedEvent object.
        """
        self._created_timestamp = self.current_timestamp

    def did_create_sell_order(self, order_created_event):
        """
        A sell order has been created. Argument is a SellOrderCreatedEvent object.
        """
        self._created_timestamp = self.current_timestamp

    def did_fill_order(self, order_filled_event):
        """
        An order has been filled in the market. Argument is a OrderFilledEvent object.
        """
        self._cancel_active_orders()
        self.logger().info(order_filled_event)
        self.shield_up(order_filled_event)

    def did_cancel_order(self, cancelled_event):
        """
        An order has been cancelled. Argument is a OrderCancelledEvent object.
        """
        self._cancel_active_orders()

    def shield_up(self, message: str):
        """
        Activate shield unless budget reallocation
        """
        if self._applied_budget_reallocation:
            self._applied_budget_reallocation = False
        else:
            self._create_timestamp = self.current_timestamp + self._filled_order_delay
            until = datetime.fromtimestamp(self._create_timestamp)
            self.notify_hb_app(
                f"{message}\nShielded up until {until} {until.astimezone().tzname()}."
            )

    def format_status(self):
        """
        Return the budget, market, miner and order statuses.
        """
        if not self._exchange_ready:
            return "Market connectors are not ready."
        lines = []
        warning_lines = []
        warning_lines.extend(self.network_warning([self._market_info]))

        # Current market data
        markets_df = map_df_to_str(self.market_status_data_frame([self._market_info]))
        lines.extend(
            ["", "  Markets:"]
            + ["    " + line for line in markets_df.to_string(index=False).split("\n")]
        )

        # Current trading balance
        wallet_df = map_df_to_str(self.wallet_balance_data_frame([self._market_info]))
        lines.extend(
            ["", "  Balance:"]
            + ["    " + line for line in wallet_df.to_string(index=False).split("\n")]
        )

        # Current active orders
        if len(self.active_orders) > 0:
            orders_df = map_df_to_str(self.active_orders_data_frame)
            lines.extend(
                ["", "  Orders:"]
                + [
                    "    " + line
                    for line in orders_df.to_string(index=False).split("\n")
                ]
            )
        else:
            lines.extend(["", "  No active maker orders."])

        warning_lines.extend(self.balance_warning([self._market_info]))
        if len(warning_lines) > 0:
            lines.extend(["", "*** WARNINGS ***"] + warning_lines)
        return "\n".join(lines)
