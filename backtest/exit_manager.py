"""
Exit Manager
============
Centralised stop-loss / take-profit logic shared across all strategies.

Usage (backtest loop)
---------------------
    from backtest.exit_manager import ExitManager, ExitResult

    em = ExitManager(direction, entry_price, sl_price, params)
    for row in bars:
        result = em.update(row)
        if result.closed:
            # handle result.reason, result.exit_price, result.pnl_frac …

Design
------
Stop layers (evaluated in priority order each bar):

1. Hard SL          – fixed price set at entry; never moves against position
2. Breakeven SL     – after Partial-TP fires, hard SL is moved to entry_price
3. Trailing SL      – ATR-based trail, activates after TRAIL_TRIGGER_R × SL_dist
                      in profit (or immediately after Partial-TP)
4. Partial TP       – closes PARTIAL_FRAC of notional at PARTIAL_R × SL_dist;
                      triggers Breakeven SL + Trail
5. Full TP          – fixed target at TP_RR × SL_dist from entry
6. Timeout          – force-exit at close after MAX_HOLD_BARS bars
7. Trend invalidation – exit at close if trend_up flips against position (optional)

All parameters are passed in via a `params` dict so the same ExitManager
works for both backtesting (grid-search) and live paper trading.

Parameter keys (all optional – defaults listed)
------------------------------------------------
SL_MODE          : "fixed_price"  | "donchian"   (default: "fixed_price")
                    fixed_price   → use the sl_price passed at construction
                    donchian      → use don_lower / don_upper from bar data

SL_MULT          : float  (default 1.5)  – only used when sl_price is computed
                    from ATR inside ExitManager (not needed if caller sets sl)

USE_PARTIAL_TP   : bool   (default True)
PARTIAL_R        : float  (default 1.0)  – partial exit at 1×initial_sl_dist
PARTIAL_FRAC     : float  (default 0.5)  – fraction of notional closed at Partial-TP

USE_TRAIL        : bool   (default True)
TRAIL_ATR_MULT   : float  (default 1.0)  – trail distance = ATR × TRAIL_ATR_MULT
TRAIL_TRIGGER_R  : float  (default 1.0)  – R-multiple profit needed to activate trail
                    (set to 0 to activate trail immediately after Partial-TP)

TP_RR            : float  (default 3.0)  – full-TP at TP_RR × initial_sl_dist
MAX_HOLD_BARS    : int    (default 0)    – 0 = disabled
USE_TREND_EXIT   : bool   (default False) – exit when trend flips
FEE_RATE         : float  (default 0.0005)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# ── Result ────────────────────────────────────────────────────────────────────
@dataclass
class PartialFill:
    exit_price: float
    frac: float          # fraction of original notional
    reason: str          # "PARTIAL_TP"
    pnl_frac: float      # pnl / notional_closed (before fee)


@dataclass
class ExitResult:
    closed: bool = False
    exit_price: float = 0.0
    reason: str = ""          # "TP" | "SL" | "TRAIL_SL" | "TIMEOUT" | "TREND_EXIT" | "VOL_DIV"
    pnl_frac: float = 0.0     # pnl per unit notional (before fee)
    partial: Optional[PartialFill] = None   # filled if partial TP fired this bar


# ── Exit Manager ──────────────────────────────────────────────────────────────
class ExitManager:
    """
    Stateful per-position exit manager.

    Parameters
    ----------
    direction   : "long" | "short"
    entry_price : float
    sl_price    : float   – initial hard stop (caller computes it)
    tp_price    : float   – full TP price (caller computes it; 0 = use TP_RR)
    params      : dict    – override any default parameter
    """

    DEFAULTS = {
        "USE_PARTIAL_TP":  True,
        "PARTIAL_R":       1.0,
        "PARTIAL_FRAC":    0.5,
        "USE_TRAIL":       True,
        "TRAIL_ATR_MULT":  1.0,
        "TRAIL_TRIGGER_R": 1.0,
        "TP_RR":           3.0,
        "MAX_HOLD_BARS":   0,
        "USE_TREND_EXIT":  False,
        "USE_VOL_DIV":     False,
        "FEE_RATE":        0.0005,
    }

    def __init__(
        self,
        direction: str,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        params: dict | None = None,
        partial_tp_price: float | None = None,
    ):
        self._p = {**self.DEFAULTS, **(params or {})}

        self.direction   = direction
        self.entry_price = entry_price
        self.sl_dist     = abs(entry_price - sl_price)   # initial SL distance
        self.bars_held   = 0

        # ── Stop ─────────────────────────────────────────────────────────────
        self.sl_price    = sl_price       # current hard stop (may move to breakeven)
        self._initial_sl = sl_price       # never changes – used as reference for R calc

        # ── Take-Profit ───────────────────────────────────────────────────────
        if tp_price and tp_price != 0.0:
            self.tp_price = tp_price
        else:
            rr = self._p["TP_RR"]
            self.tp_price = (entry_price + self.sl_dist * rr if direction == "long"
                             else entry_price - self.sl_dist * rr)

        # ── Partial TP ────────────────────────────────────────────────────────
        if partial_tp_price is not None:
            self.partial_tp_price = partial_tp_price
        else:
            r = self._p["PARTIAL_R"]
            self.partial_tp_price = (entry_price + self.sl_dist * r if direction == "long"
                                     else entry_price - self.sl_dist * r)
        self.partial_done = False

        # ── Trail ─────────────────────────────────────────────────────────────
        self.trail_active = False
        self._trail_ref   = sl_price      # trail reference; updated each bar

    # ── Public ────────────────────────────────────────────────────────────────
    def update(self, row) -> ExitResult:
        """
        Process one bar (a pandas Series or dict with keys:
        high, low, close, atr, trend_up [optional]).
        Returns an ExitResult. Mutates internal state.
        """
        self.bars_held += 1
        result = ExitResult()

        # ── 1. Try Partial TP ────────────────────────────────────────────────
        if self._p["USE_PARTIAL_TP"] and not self.partial_done:
            partial = self._check_partial(row)
            if partial:
                result.partial = partial
                self._on_partial_tp()

        # ── 2. Update trailing stop ──────────────────────────────────────────
        if self.trail_active or (self._p["USE_TRAIL"] and not self.partial_done):
            self._update_trail(row)

        # ── 3. Check timeout ─────────────────────────────────────────────────
        max_hold = self._p["MAX_HOLD_BARS"]
        if max_hold > 0 and self.bars_held >= max_hold:
            result.closed     = True
            result.exit_price = float(row["close"])
            result.reason     = "TIMEOUT"
            result.pnl_frac   = self._pnl_frac(float(row["close"]))
            return result

        # ── 4. Check trend-exit ──────────────────────────────────────────────
        if self._p["USE_TREND_EXIT"] and "trend_up" in row:
            flipped = (self.direction == "long"  and not row["trend_up"]) or \
                      (self.direction == "short" and     row["trend_up"])
            if flipped:
                result.closed     = True
                result.exit_price = float(row["close"])
                result.reason     = "TREND_EXIT"
                result.pnl_frac   = self._pnl_frac(float(row["close"]))
                return result

        # ── 5. Check vol_div exit ────────────────────────────────────────────
        if self._p["USE_VOL_DIV"]:
            vd_key = "vol_div_long" if self.direction == "long" else "vol_div_short"
            if row.get(vd_key, False):
                result.closed     = True
                result.exit_price = float(row["close"])
                result.reason     = "VOL_DIV"
                result.pnl_frac   = self._pnl_frac(float(row["close"]))
                return result

        # ── 6. Check TP / SL ─────────────────────────────────────────────────
        hit_tp, hit_sl = self._check_tp_sl(row)

        if hit_tp:
            result.closed     = True
            result.exit_price = self.tp_price
            result.reason     = "TP"
            result.pnl_frac   = self._pnl_frac(self.tp_price)
        elif hit_sl:
            result.closed     = True
            result.exit_price = self.sl_price
            result.reason     = "TRAIL_SL" if self.trail_active else "SL"
            result.pnl_frac   = self._pnl_frac(self.sl_price)

        return result

    # ── Properties ────────────────────────────────────────────────────────────
    @property
    def current_sl(self) -> float:
        return self.sl_price

    @property
    def current_tp(self) -> float:
        return self.tp_price

    def r_multiple(self, price: float) -> float:
        """Unrealized R-multiple at given price."""
        dist = (price - self.entry_price if self.direction == "long"
                else self.entry_price - price)
        return dist / self.sl_dist if self.sl_dist > 0 else 0.0

    # ── Internal ──────────────────────────────────────────────────────────────
    def _pnl_frac(self, exit_price: float) -> float:
        if self.direction == "long":
            return (exit_price - self.entry_price) / self.entry_price
        return (self.entry_price - exit_price) / self.entry_price

    def _check_partial(self, row) -> Optional[PartialFill]:
        ptp = self.partial_tp_price
        hit = (float(row["high"]) >= ptp if self.direction == "long"
               else float(row["low"]) <= ptp)
        if not hit:
            return None
        frac = self._p["PARTIAL_FRAC"]
        pnl  = self._pnl_frac(ptp)
        return PartialFill(exit_price=ptp, frac=frac, reason="PARTIAL_TP", pnl_frac=pnl)

    def _on_partial_tp(self):
        """State changes triggered by partial TP."""
        self.partial_done = True
        # Move SL to breakeven
        self.sl_price     = self.entry_price
        # Activate trail immediately from breakeven
        self.trail_active = True
        self._trail_ref   = self.entry_price

    def _update_trail(self, row):
        """Ratchet trail stop. Only advances in the profitable direction."""
        atr  = float(row.get("atr", 0)) or 0.0
        mult = self._p["TRAIL_ATR_MULT"]
        trig = self._p["TRAIL_TRIGGER_R"]

        if not self.trail_active:
            # Check if profit has reached trigger level
            profit_r = self.r_multiple(float(row["close"]))
            if profit_r >= trig:
                self.trail_active = True

        if self.trail_active and atr > 0:
            if self.direction == "long":
                new_sl = float(row["low"]) - atr * mult
                self.sl_price = max(self.sl_price, new_sl)
            else:
                new_sl = float(row["high"]) + atr * mult
                self.sl_price = min(self.sl_price, new_sl)

    def _check_tp_sl(self, row):
        hit_tp = (float(row["high"]) >= self.tp_price if self.direction == "long"
                  else float(row["low"])  <= self.tp_price)
        hit_sl = (float(row["low"])  <= self.sl_price if self.direction == "long"
                  else float(row["high"]) >= self.sl_price)
        return hit_tp, hit_sl
