from datetime import datetime

from hummingbot.pmm_script.pmm_script_base import PMMScriptBase


class HelloWorldPMMScript(PMMScriptBase):
    """
    Demonstrates how to send messages using notify and log functions. It also shows how errors and commands are handled.
    """

    def on_tick(self):
        tick = datetime.now().second
        self.log(f"Current tick: {tick}")

        if tick >= 57 or tick <= 3:
            self.pmm_parameters.buy_levels = self.pmm_parameters.order_levels
            self.pmm_parameters.sell_levels = self.pmm_parameters.order_levels

        if 3 < tick < 57:
            self.pmm_parameters.buy_levels = 0
            self.pmm_parameters.sell_levels = 0
