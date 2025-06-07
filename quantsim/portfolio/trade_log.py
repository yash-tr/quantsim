"""Manages detailed fill records and reconstructs round-trip trades.

This module provides `DetailedFill` to store data from individual fill events,
and the `Trade` dataclass to represent a complete round-trip trade (or an
open position being built). The `TradeLog` class processes `FillEvent`s
to build these `Trade` objects, handling scenarios like scaling in/out of
positions and position flips based on a simplified FIFO-like approach per symbol.
"""

import csv
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import List, Dict, Optional, NamedTuple
from quantsim.core.events import FillEvent


class DetailedFill(NamedTuple):
    """Stores attributes of a single fill event for record-keeping.

    Attributes:
        timestamp (datetime): Time of the fill.
        symbol (str): Symbol traded.
        order_id (Optional[str]): ID of the order that led to this fill.
        direction (str): Direction of the fill ('BUY' or 'SELL').
        quantity (float): Quantity filled.
        fill_price (float): Price at which the fill occurred.
        commission (float): Commission paid for this fill.
        exchange (Optional[str]): Exchange where fill occurred.
    """

    timestamp: datetime
    symbol: str
    order_id: Optional[str]
    direction: str
    quantity: float
    fill_price: float
    commission: float
    exchange: Optional[str]


@dataclass
class Trade:
    """Represents a single round-trip trade or an open position being built.

    A trade is initiated by a fill and is considered open until an opposing fill
    (or series of fills) closes out the initial quantity for that entry direction.
    This class accumulates details of entry and exit fills and calculates
    various metrics for the trade.

    Attributes:
        symbol (str): The symbol traded.
        entry_timestamp (datetime): Timestamp of the first fill that initiated this trade.
        direction (str): Direction of the initial entry ('LONG' or 'SHORT').

        quantity_total_entry (float): Total quantity accumulated for the entry leg of this trade.
        value_total_entry (float): Sum of (price * quantity) for all entry fills.
        total_entry_commission (float): Sum of commissions for all entry fills.

        quantity_total_exit (float): Total quantity accumulated for the exit leg of this trade.
        value_total_exit (float): Sum of (price * quantity) for all exit fills.
        total_exit_commission (float): Sum of commissions for all exit fills.

        quantity_open (float): Current net open quantity for this trade instance.
                               Decreases as exit fills are processed.
        is_open (bool): True if the trade instance still has an open quantity.
        exit_timestamp (Optional[datetime]): Timestamp of the last fill that closed or
                                             reduced this specific trade instance.

        realized_pnl (float): Gross profit or loss from the closed portions of this trade
                              (i.e., (exit_price - entry_price) * quantity, adjusted for direction).

        entry_fills (List[DetailedFill]): List of `DetailedFill` objects for the entry leg.
        exit_fills (List[DetailedFill]): List of `DetailedFill` objects for the exit leg.
    """

    symbol: str
    entry_timestamp: datetime
    direction: str

    quantity_total_entry: float = 0.0
    value_total_entry: float = 0.0
    total_entry_commission: float = 0.0

    quantity_total_exit: float = 0.0
    value_total_exit: float = 0.0
    total_exit_commission: float = 0.0

    quantity_open: float = 0.0
    is_open: bool = True
    exit_timestamp: Optional[datetime] = None

    realized_pnl: float = 0.0

    entry_fills: List[DetailedFill] = field(default_factory=list)
    exit_fills: List[DetailedFill] = field(default_factory=list)

    @property
    def avg_entry_price(self) -> float:
        """float: Average price of all entry fills for this trade."""
        return (
            self.value_total_entry / self.quantity_total_entry
            if self.quantity_total_entry > 1e-9
            else 0.0
        )

    @property
    def avg_exit_price(self) -> float:
        """float: Average price of all exit fills for this trade."""
        return (
            self.value_total_exit / self.quantity_total_exit
            if self.quantity_total_exit > 1e-9
            else 0.0
        )

    @property
    def duration_seconds(self) -> Optional[float]:
        """Optional[float]: Duration of the trade in seconds if closed, else None."""
        if not self.is_open and self.entry_timestamp and self.exit_timestamp:
            return (self.exit_timestamp - self.entry_timestamp).total_seconds()
        return None

    @property
    def net_pnl(self) -> float:
        """float: Net profit or loss for this trade (realized_pnl less total commissions)."""
        return self.realized_pnl - (
            self.total_entry_commission + self.total_exit_commission
        )

    def __repr__(self) -> str:
        state = "OPEN" if self.is_open else "CLOSED"
        return (
            f"Trade(Sym:{self.symbol}, Dir:{self.direction}, St:{state}, QOpen:{self.quantity_open:.2f}, "
            f"AvgEntry:{self.avg_entry_price:.2f}, AvgExit:{self.avg_exit_price:.2f}, NetPnL:{self.net_pnl:.2f})"
        )


class TradeLog:
    """Processes fill events to reconstruct and log round-trip trades.

    This log maintains a list of completed trades and tracks currently open
    trade instances for each symbol. It uses a simplified model where, for each
    symbol, it considers one "active trade instance" at a time. Fills matching
    the direction of this instance scale it in. Opposite fills reduce or close it.
    If an opposite fill exceeds the open quantity, the existing trade is closed,
    and a new trade instance is opened in the new direction (a flip).

    Attributes:
        completed_trades (List[Trade]): A list of all trades that have been fully closed.
        open_trade_per_symbol (Dict[str, Trade]): A dictionary mapping symbols to their
            currently open `Trade` instance.
    """

    def __init__(self):
        """Initializes the TradeLog with empty lists for trades."""
        self.completed_trades: List[Trade] = []
        self.open_trade_per_symbol: Dict[str, Trade] = {}

    def _create_detailed_fill(self, fill_event: FillEvent) -> DetailedFill:
        """Converts a FillEvent to a DetailedFill named tuple.

        Args:
            fill_event (FillEvent): The input fill event.

        Returns:
            DetailedFill: A named tuple containing relevant details from the fill.
        """
        return DetailedFill(
            timestamp=fill_event.timestamp,
            symbol=fill_event.symbol,
            order_id=fill_event.order_id,
            direction=fill_event.direction,
            quantity=fill_event.quantity,
            fill_price=fill_event.fill_price,
            commission=fill_event.commission,
            exchange=fill_event.exchange,
        )

    def process_fill(self, fill_event: FillEvent) -> None:
        """Processes a single `FillEvent` to update trade records.

        This method identifies whether the fill opens a new trade, scales an
        existing one, closes/reduces an existing one, or flips a position.
        It updates the `open_trade_per_symbol` dictionary and appends to
        `completed_trades` as trades are closed.

        Args:
            fill_event (FillEvent): The fill event to process.
        """
        symbol = fill_event.symbol
        detailed_fill = self._create_detailed_fill(fill_event)
        open_trade = self.open_trade_per_symbol.get(symbol)

        fill_qty = fill_event.quantity
        fill_px = fill_event.fill_price
        fill_comm = fill_event.commission

        if not open_trade:
            # No open trade for this symbol, this fill starts a new one
            new_trade = Trade(
                symbol=symbol,
                entry_timestamp=fill_event.timestamp,
                direction=fill_event.direction,
                quantity_total_entry=fill_qty,
                value_total_entry=(fill_px * fill_qty),
                total_entry_commission=fill_comm,
                quantity_open=fill_qty,
                is_open=True,
            )
            new_trade.entry_fills.append(detailed_fill)
            self.open_trade_per_symbol[symbol] = new_trade
        elif open_trade.direction == fill_event.direction:
            # Fill is same direction: scaling into existing open trade
            open_trade.value_total_entry += fill_px * fill_qty
            open_trade.quantity_total_entry += fill_qty
            open_trade.total_entry_commission += fill_comm
            open_trade.quantity_open += fill_qty
            open_trade.entry_fills.append(detailed_fill)
        else:
            # Fill is opposite to open trade: closing, reducing, or flipping
            qty_closed_on_this_fill = min(fill_qty, open_trade.quantity_open)

            # Update exit accumulators for the open trade being closed/reduced
            open_trade.value_total_exit += fill_px * qty_closed_on_this_fill
            open_trade.quantity_total_exit += qty_closed_on_this_fill
            commission_for_this_close_portion = (
                fill_comm * (qty_closed_on_this_fill / fill_qty)
                if fill_qty > 1e-9
                else 0
            )
            open_trade.total_exit_commission += commission_for_this_close_portion

            # Store the part of the fill that applies to closing this trade
            # Create a new DetailedFill for the exit_fills list with the exact quantity used for this closing part
            closing_part_fill = detailed_fill._replace(
                quantity=qty_closed_on_this_fill,
                commission=commission_for_this_close_portion,
            )
            open_trade.exit_fills.append(closing_part_fill)

            # Calculate PnL for this closed portion
            if open_trade.direction == "LONG":
                pnl = (fill_px - open_trade.avg_entry_price) * qty_closed_on_this_fill
            else:  # SHORT
                pnl = (open_trade.avg_entry_price - fill_px) * qty_closed_on_this_fill
            open_trade.realized_pnl += pnl

            open_trade.quantity_open -= qty_closed_on_this_fill
            open_trade.exit_timestamp = (
                fill_event.timestamp
            )  # Update with latest exit activity

            if open_trade.quantity_open <= 1e-9:  # Trade instance fully closed
                open_trade.is_open = False
                open_trade.quantity_open = 0.0  # Clean up small float residuals
                self.completed_trades.append(open_trade)
                del self.open_trade_per_symbol[symbol]

            # Handle flip: if fill quantity was greater than what was needed to close
            qty_remaining_in_fill_for_flip = fill_qty - qty_closed_on_this_fill
            if qty_remaining_in_fill_for_flip > 1e-9:
                commission_for_flipped_part = (
                    fill_comm * (qty_remaining_in_fill_for_flip / fill_qty)
                    if fill_qty > 1e-9
                    else 0
                )
                flipped_trade = Trade(
                    symbol=symbol,
                    entry_timestamp=fill_event.timestamp,
                    direction=fill_event.direction,
                    quantity_total_entry=qty_remaining_in_fill_for_flip,
                    value_total_entry=(fill_px * qty_remaining_in_fill_for_flip),
                    total_entry_commission=commission_for_flipped_part,
                    quantity_open=qty_remaining_in_fill_for_flip,
                    is_open=True,
                )
                # The remaining part of the original detailed_fill opens the new trade
                flipped_trade.entry_fills.append(
                    detailed_fill._replace(
                        quantity=qty_remaining_in_fill_for_flip,
                        commission=commission_for_flipped_part,
                    )
                )
                self.open_trade_per_symbol[symbol] = flipped_trade

    def get_completed_trades(self) -> List[Trade]:
        """Returns all completed trades."""
        return self.completed_trades.copy()

    def add_fill(self, fill_event: FillEvent) -> None:
        """Add a fill event to the trade log.

        This is an alias for process_fill to maintain compatibility.

        Args:
            fill_event: The fill event to process
        """
        self.process_fill(fill_event)

    def get_open_trades(self) -> List[Trade]:
        """Returns all currently open trades."""
        return list(self.open_trade_per_symbol.values())

    def to_csv(self, filepath: str, include_open_trades: bool = False) -> None:
        """Exports trades to a CSV file.

        Args:
            filepath (str): The path to the CSV file to be created.
            include_open_trades (bool, optional): If True, currently open trades
                will also be included in the export. Defaults to False.
        """
        trades_to_export = list(self.completed_trades)  # Start with a copy
        if include_open_trades:
            trades_to_export.extend(self.get_open_trades())

        if not trades_to_export:
            print("TradeLog: No trades to export.")
            return

        print(f"TradeLog: Exporting {len(trades_to_export)} trades to {filepath}...")
        try:
            with open(filepath, "w", newline="") as csvfile:
                # Dynamically get field names from the Trade dataclass, excluding lists of fills
                header = [
                    f.name
                    for f in fields(Trade)
                    if f.name not in ["entry_fills", "exit_fills"]
                ]
                # Add property-based fields to header
                header.extend(
                    ["avg_entry_price", "avg_exit_price", "net_pnl", "duration_seconds"]
                )

                writer = csv.DictWriter(
                    csvfile, fieldnames=header, extrasaction="ignore"
                )
                writer.writeheader()
                for trade in trades_to_export:
                    row_data = {}
                    # Populate direct attributes
                    for f_name in header:  # Iterate through desired header fields
                        if hasattr(trade, f_name) and f_name not in [
                            "avg_entry_price",
                            "avg_exit_price",
                            "net_pnl",
                            "duration_seconds",
                        ]:
                            row_data[f_name] = getattr(trade, f_name)
                    # Populate properties
                    row_data["avg_entry_price"] = trade.avg_entry_price
                    row_data["avg_exit_price"] = trade.avg_exit_price
                    row_data["net_pnl"] = trade.net_pnl
                    duration = trade.duration_seconds
                    row_data["duration_seconds"] = (
                        duration if duration is not None else ""
                    )  # CSV friendly

                    writer.writerow(row_data)
            print(f"TradeLog: Successfully exported trades to {filepath}.")
        except IOError as e:
            print(f"TradeLog: Error exporting trades to CSV - {e}")
