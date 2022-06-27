from datetime import datetime
from decimal import Decimal

from hummingbot.pmm_script.pmm_script_base import PMMScriptBase

MIN_VOLATILITY_PCT = Decimal("0.2")  # in percentage
ORDERS_DELAY = 60  # in seconds


class OrdersAdjustedOnVolatilityScript(PMMScriptBase):
    _create_timestamp = 0

    def on_tick(self):
        if self.mid_price_change_pct > MIN_VOLATILITY_PCT:
            self._create_timestamp = self.current_timestamp + ORDERS_DELAY
            self.log(self.volatility_msg)
            self.notify(self.volatility_msg)

        if self.current_timestamp < self._create_timestamp:
            self.pmm_parameters.buy_levels = 0
            self.pmm_parameters.sell_levels = 0
        else:
            self.pmm_parameters.buy_levels = self.pmm_parameters.order_levels
            self.pmm_parameters.sell_levels = self.pmm_parameters.order_levels

    @property
    def current_timestamp(self):
        return datetime.utcnow().timestamp()

    @property
    def mid_price_change(self):
        if len(self.mid_prices) >= 2:
            return self.mid_prices[-1] / self.mid_prices[-2] - Decimal("1")
        return 0

    @property
    def mid_price_change_pct(self):
        return abs(self.mid_price_change * Decimal("100"))

    @property
    def volatility_msg(self):
        until = datetime.fromtimestamp(self._create_timestamp).astimezone()
        return (
            f"Delay orders creation until {until} {until.tzname()}. "
            f"mid_price: {self.mid_price} change: {self.mid_price_change_pct:.2f}%."
        )
