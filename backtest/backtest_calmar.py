"""
Calmar-Optimised Trend Breakout  (large-capital / robust variant)
==================================================================
Entry   : 1h close breaks Donchian upper/lower  (same as backtest_breakout.py)
Trend   : 1d close > EMA(TREND_EMA_PERIOD)
Filters : ADX > ADX_MIN  +  volume spike
SL      : entry ± ATR × SL_MULT
TP      : entry ± ATR × SL_MULT × TP_RR   OR  trailing stop

Key differences from backtest_breakout.py
──────────────────────────────────────────
1. OPTIMIZE_TARGET = "calmar"   → rank combos by Calmar (CAGR/MaxDD)
                                  instead of raw total-return
2. Volatility-targeting size    → notional = capital × VOL_TARGET / realised_ann_vol
   (USE_VOL_TARGET = True)         adapts position size to current market vol;
                                  standard in CTA / quant macro funds
3. Conservative leverage        → TUNE_SPACE uses [2, 3, 5] instead of [10, 20]
4. MIN_TRADE_COUNT guard        → combos with < 30 trades are disqualified
                                  (prevents over-fit single lucky trades)
5. Separate results dir         → results/calmar/
"""

# ── Must be set BEFORE numpy/pandas import to prevent fork deadlock ───────────
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

_ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR    = _ROOT / "data"
RESULTS_DIR = _ROOT / "results/calmar"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RAW_DATA: dict = {}
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
FEE_RATE         = 0.0005      # 0.05 % per side (taker)
LEVERAGE         = 3           # conservative cap; vol-targeting may use less

BASE_RISK        = 0.01        # fallback risk % when vol-targeting is off
DONCHIAN_PERIOD  = 20
ATR_PERIOD       = 14
SL_MULT          = 1.5
TP_RR            = 3.0
TREND_EMA_PERIOD = 200

# ── Volatility targeting ──────────────────────────────────────────────────────
USE_VOL_TARGET  = True         # True = size by vol target; False = fixed BASE_RISK
VOL_TARGET      = 0.20         # target annual portfolio volatility (20 %)
VOL_LOOKBACK    = 24           # hours of log-returns to estimate realised vol

# ── Breakout quality filters ──────────────────────────────────────────────────
ADX_PERIOD    = 14
ADX_MIN       = 25.0           # 0 = disabled
VOL_MA_PERIOD = 20
VOL_MULT      = 1.5            # 1.0 = disabled

# ── Trailing stop ─────────────────────────────────────────────────────────────
USE_TRAIL       = False
TRAIL_TRIGGER_R = 1.0
TRAIL_MULT      = 1.0

# ── Re-entry cooldown ─────────────────────────────────────────────────────────
COOLDOWN_BARS = 0

# ── Partial TP (split exit) ───────────────────────────────────────────────────
USE_PARTIAL_TP   = True    # True = close 50 % at +1R, trail the rest
PARTIAL_TP_R     = 1.0     # first exit at entry + SL_dist × PARTIAL_TP_R
PARTIAL_TP_FRAC  = 0.5     # fraction of notional to close at first TP

# ── Pullback entry ────────────────────────────────────────────────────────────
USE_PULLBACK     = False   # True = wait for price to retrace before entering
PULLBACK_ATR     = 0.5     # enter when price pulls back within X×ATR of level
PULLBACK_WINDOW  = 6       # bars to wait for the pullback

# ── Time-based exit ───────────────────────────────────────────────────────────
MAX_HOLD_BARS    = 0       # 0 = disabled; e.g. 72 = close after 72 bars

# ── ADX slope filter ─────────────────────────────────────────────────────────
ADX_SLOPE_BARS   = 3       # require ADX rising over this many bars (0=disabled)

# ── Max simultaneous open positions ──────────────────────────────────────────
MAX_OPEN_POS     = 1       # per-coin backtest always has 1 coin, kept for portfolio use

# ── Auto-tuning ───────────────────────────────────────────────────────────────
AUTO_TUNE        = True
OPTIMIZE_TARGET  = "calmar"    # "calmar" | "sharpe" | "return"
MIN_TRADE_COUNT  = 30          # disqualify combos with fewer trades (anti-overfit)

TUNE_SPACE = {
    # Narrowed based on best_params.json: keep only values that appeared as optimal
    "LEVERAGE":          [2, 5],           # 3 never appeared in best
    "USE_VOL_TARGET":    [True, False],
    "VOL_TARGET":        [0.30],           # all best were 0.30
    "DONCHIAN_PERIOD":   [10, 20, 40],
    "ATR_PERIOD":        [7, 14],
    "SL_MULT":           [1.0, 1.5, 2.0],
    "TP_RR":             [3.0, 5.0],       # 2.0 never appeared in best
    "TREND_EMA_PERIOD":  [50, 200],        # 100 never appeared in best
    "ADX_MIN":           [0.0, 20.0],      # 25.0 never appeared in best
    "ADX_SLOPE_BARS":    [0, 3],
    "VOL_MULT":          [1.0, 1.5],
    "COOLDOWN_BARS":     [0, 3],
    "USE_PARTIAL_TP":    [True, False],
    "PARTIAL_TP_R":      [1.0, 1.5],
    "USE_PULLBACK":      [True, False],
    "MAX_HOLD_BARS":     [0, 72],
}
# Total: 2×2×1×3×2×3×2×2×2×2×2×2×2×2×2×3 = 110,592 combinations (vs 1.68M before)

COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
    ("SUI/USDT:USDT", "sui"),
]


# ── Indicators ────────────────────────────────────────────────────────────────
def prepare(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()

    # Donchian channel (shift to avoid look-ahead)
    df["don_upper"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["don_lower"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    # ATR (EWM-smoothed)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # Realised volatility for vol-targeting  (annualised from 1h log-returns)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realised_vol"] = log_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(24 * 365)

    # ADX (Wilder-style EWM)
    up_move  = df["high"] - df["high"].shift(1)
    dn_move  = df["low"].shift(1) - df["low"]
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_dx   = tr.ewm(span=ADX_PERIOD, adjust=False).mean().clip(lower=1e-9)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9)
    df["adx"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

    # Volume spike confirmation
    if "volume" in df.columns:
        df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
        df["vol_ok"] = df["volume"] >= df["vol_ma"] * VOL_MULT
    else:
        df["vol_ok"] = True

    # ADX slope: rising over last ADX_SLOPE_BARS bars
    if ADX_SLOPE_BARS > 0:
        df["adx_slope_ok"] = df["adx"] > df["adx"].shift(ADX_SLOPE_BARS)
    else:
        df["adx_slope_ok"] = True

    # Breakout signals
    df["entry_long"]  = df["close"] > df["don_upper"]
    df["entry_short"] = df["close"] < df["don_lower"]

    # 1d trend filter
    d1     = df_1d.copy()
    d1_ema = d1["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    d1["trend_up"] = d1["close"] > d1_ema
    trend  = d1["trend_up"].reindex(df.index, method="ffill").ffill()
    df["trend_up"] = np.where(trend.isna(), False, trend).astype(bool)

    return df


# ── Metrics & display helpers ─────────────────────────────────────────────────
def compute_metrics(t: pd.DataFrame, initial_capital: float) -> dict:
    n        = len(t)
    wins     = int((t["pnl_usdt"] > 0).sum())
    losses   = n - wins
    win_rate = wins / n * 100
    avg_win  = float(t.loc[t["pnl_usdt"] > 0,  "pnl_usdt"].mean()) if wins   else 0.0
    avg_loss = float(t.loc[t["pnl_usdt"] <= 0, "pnl_usdt"].mean()) if losses else 0.0
    pf       = (wins * avg_win / (-losses * avg_loss)
                if losses and avg_loss < 0 else float("inf"))
    expectancy = float(t["pnl_usdt"].mean())
    final      = float(t["capital"].iloc[-1])
    total_ret  = (final - initial_capital) / initial_capital * 100
    max_dd     = float(t["drawdown"].max()) * 100

    eq        = t.set_index("exit_time")["capital"].resample("1D").last().ffill()
    first_day = eq.index[0] - pd.Timedelta(days=1)
    eq        = pd.concat([pd.Series({first_day: float(initial_capital)}), eq])
    daily_ret = eq.pct_change().dropna()
    sharpe    = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    span_days = max((t["exit_time"].iloc[-1] - t["exit_time"].iloc[0]).days, 1)
    ann_ret   = (final / initial_capital) ** (365.25 / span_days) - 1
    calmar    = float(ann_ret / (max_dd / 100)) if max_dd > 0 else float("inf")

    return dict(
        n=n, wins=wins, losses=losses, win_rate=win_rate,
        avg_win=avg_win, avg_loss=avg_loss, pf=pf, expectancy=expectancy,
        total_ret=total_ret, max_dd=max_dd, sharpe=sharpe, calmar=calmar,
        ann_ret=ann_ret * 100, final=final,
    )


def _score(m: dict) -> float:
    """Scalar ranking score for a single coin's metrics dict."""
    if m is None or m["n"] < MIN_TRADE_COUNT:
        return float("-inf")
    if OPTIMIZE_TARGET == "calmar":
        v = m["calmar"]
        return v if v < 1e6 else 0.0   # guard: infinite calmar = zero drawdown = suspect
    if OPTIMIZE_TARGET == "sharpe":
        return m["sharpe"]
    return m["total_ret"] / 100        # "return" mode


def print_summary_table(strategy_name: str, header: str, metrics: dict):
    cols   = ["Coin",  "Trades", "Win%",  "AvgWin$", "AvgLoss$", "PF",
              "Expect$", "Ann%",   "MaxDD%", "Sharpe",  "Calmar"]
    widths = [5,        7,        6,        9,         9,          6,
              8,         7,        7,        7,         7]

    def _sep(l, m, r): return l + m.join("─" * (w + 2) for w in widths) + r
    def _row(vals):    return "|" + "|".join(f" {str(v):>{w}} " for v, w in zip(vals, widths)) + "|"

    total_w = sum(w + 3 for w in widths) + 1
    title   = f" {strategy_name}  *  {header} "
    print(f"\n+{'-' * (total_w - 2)}+")
    print(f"|{title:<{total_w - 2}}|")
    print(_sep("+", "+", "+"))
    print(_row(cols))
    print(_sep("+", "+", "+"))
    for coin, m in metrics.items():
        pf_s  = f"{m['pf']:.2f}"     if m["pf"]     < 999 else "inf"
        cal_s = f"{m['calmar']:.2f}" if m["calmar"]  < 999 else "inf"
        print(_row([
            coin.upper(), m["n"],
            f"{m['win_rate']:.1f}%", f"${m['avg_win']:.1f}", f"${m['avg_loss']:.1f}",
            pf_s, f"${m['expectancy']:.1f}",
            f"{m['ann_ret']:.1f}%", f"{m['max_dd']:.1f}%",
            f"{m['sharpe']:.2f}", cal_s,
        ]))
    print(_sep("+", "+", "+"))


# ── Data loading ──────────────────────────────────────────────────────────────
def preload_data():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
        )


# ── Single-coin backtest ──────────────────────────────────────────────────────
def run_backtest(symbol: str, coin: str):
    print(f"\n{'='*50}")
    print(f"  {symbol}")
    print(f"{'='*50}")

    df_1h, df_1d = _RAW_DATA.get(coin) or (
        pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
        pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
    )
    df = prepare(df_1h, df_1d)

    capital  = float(INITIAL_CAPITAL)
    peak_cap = capital
    trades   = []

    in_trade           = False
    direction          = None
    entry_price        = 0.0
    sl_price           = 0.0
    tp_price           = 0.0      # full TP (used when partial TP already taken or disabled)
    partial_tp_price   = 0.0      # first partial-TP level
    partial_done       = False    # whether the first partial exit has been taken
    notional           = 0.0      # remaining notional after possible partial exit
    notional_full      = 0.0      # original notional at entry
    peak_loss_ratio    = 0.0
    trail_active       = False
    trail_sl           = 0.0
    cooldown_remaining = 0
    bars_in_trade      = 0

    # pullback-pending state
    pb_pending    = False         # True = breakout seen, waiting for pullback entry
    pb_direction  = None
    pb_level      = 0.0           # don_upper / don_lower at signal bar
    pb_atr        = 0.0
    pb_bars_left  = 0

    warmup = max(DONCHIAN_PERIOD, ATR_PERIOD, ADX_PERIOD, VOL_MA_PERIOD,
                 VOL_LOOKBACK, ADX_SLOPE_BARS) + 2

    _iter = tqdm(range(warmup, len(df)), desc=f"{coin.upper()}",
                 unit="bar", file=sys.stdout,
                 disable=not sys.stdout.isatty(), dynamic_ncols=True)

    for i in _iter:
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        # ── Exit ─────────────────────────────────────────────────────────────
        if in_trade:
            bars_in_trade += 1

            # update trailing stop
            if USE_TRAIL or (USE_PARTIAL_TP and partial_done):
                profit_dist = ((row["close"] - entry_price) if direction == "long"
                               else (entry_price - row["close"]))
                sl_dist_abs = abs(entry_price - trail_sl)
                if not trail_active and profit_dist >= TRAIL_TRIGGER_R * sl_dist_abs:
                    trail_active = True
                if trail_active:
                    atr_cur = row["atr"]
                    if direction == "long":
                        sl_price = max(sl_price, row["low"] - atr_cur * TRAIL_MULT)
                    else:
                        sl_price = min(sl_price, row["high"] + atr_cur * TRAIL_MULT)

            # ── Partial TP: first exit ────────────────────────────────────────
            if USE_PARTIAL_TP and not partial_done:
                hit_partial = (row["high"] >= partial_tp_price if direction == "long"
                               else row["low"] <= partial_tp_price)
                if hit_partial:
                    close_notional = notional_full * PARTIAL_TP_FRAC
                    pct_p = ((partial_tp_price - entry_price) / entry_price if direction == "long"
                             else (entry_price - partial_tp_price) / entry_price)
                    pnl_p = close_notional * pct_p - close_notional * FEE_RATE * 2
                    capital  += pnl_p
                    peak_cap  = max(peak_cap, capital)
                    notional  = notional_full * (1.0 - PARTIAL_TP_FRAC)
                    partial_done = True
                    trail_active = True   # immediately start trailing remainder
                    trail_sl     = sl_price
                    trades.append({
                        "exit_time":       ts,
                        "direction":       direction,
                        "entry_price":     round(entry_price, 6),
                        "exit_price":      round(partial_tp_price, 6),
                        "notional":        round(close_notional, 4),
                        "exit_reason":     "PARTIAL_TP",
                        "peak_loss_ratio": round(peak_loss_ratio, 6),
                        "pnl_usdt":        round(pnl_p, 4),
                        "capital":         round(capital, 4),
                        "drawdown":        round((peak_cap - capital) / peak_cap, 6),
                    })
                    peak_loss_ratio = 0.0

            # ── Full exit: TP / SL / time ─────────────────────────────────────
            hit_tp   = (row["high"] >= tp_price if direction == "long" else row["low"]  <= tp_price)
            hit_sl   = (row["low"]  <= sl_price if direction == "long" else row["high"] >= sl_price)
            hit_time = (MAX_HOLD_BARS > 0 and bars_in_trade >= MAX_HOLD_BARS)

            worst_pnl = ((row["low"]  - entry_price) / entry_price * notional if direction == "long"
                         else (entry_price - row["high"]) / entry_price * notional)
            if worst_pnl < 0:
                peak_loss_ratio = max(peak_loss_ratio, -worst_pnl / capital)

            if hit_tp or hit_sl or hit_time:
                if hit_tp:
                    exit_price  = tp_price
                    exit_reason = "TP"
                elif hit_sl:
                    exit_price  = sl_price
                    exit_reason = "SL"
                else:
                    exit_price  = row["close"]
                    exit_reason = "TIME"

                pct = ((exit_price - entry_price) / entry_price if direction == "long"
                       else (entry_price - exit_price) / entry_price)
                pnl = notional * pct - notional * FEE_RATE * 2
                pnl = max(pnl, -capital)
                capital  += pnl
                peak_cap  = max(peak_cap, capital)
                trades.append({
                    "exit_time":       ts,
                    "direction":       direction,
                    "entry_price":     round(entry_price, 6),
                    "exit_price":      round(exit_price, 6),
                    "notional":        round(notional, 4),
                    "exit_reason":     exit_reason,
                    "peak_loss_ratio": round(peak_loss_ratio, 6),
                    "pnl_usdt":        round(pnl, 4),
                    "capital":         round(capital, 4),
                    "drawdown":        round((peak_cap - capital) / peak_cap, 6),
                })
                peak_loss_ratio    = 0.0
                trail_active       = False
                trail_sl           = 0.0
                partial_done       = False
                bars_in_trade      = 0
                in_trade           = False
                if exit_reason == "SL":
                    cooldown_remaining = COOLDOWN_BARS

        # ── Pullback-pending: check for retrace entry ─────────────────────────
        if not in_trade and pb_pending:
            pb_bars_left -= 1
            entered = False
            if pb_direction == "long":
                target = pb_level - pb_atr * PULLBACK_ATR
                if row["low"] <= pb_level and row["close"] >= target:
                    entry_price = max(row["close"], target)
                    entered = True
            else:
                target = pb_level + pb_atr * PULLBACK_ATR
                if row["high"] >= pb_level and row["close"] <= target:
                    entry_price = min(row["close"], target)
                    entered = True

            if entered:
                direction    = pb_direction
                pb_pending   = False
                atr          = row["atr"]
                sl_dist      = atr * SL_MULT
                sl_dist_pct  = sl_dist / entry_price
                if USE_VOL_TARGET:
                    rv = row["realised_vol"]
                    notional_full = (capital * VOL_TARGET / rv
                                     if not pd.isna(rv) and rv > 1e-6
                                     else capital * BASE_RISK / sl_dist_pct)
                else:
                    notional_full = capital * BASE_RISK / sl_dist_pct
                notional_full = min(notional_full, capital * LEVERAGE)
                notional      = notional_full
                sl_price      = (entry_price - sl_dist if direction == "long"
                                 else entry_price + sl_dist)
                tp_price      = (entry_price + sl_dist * TP_RR if direction == "long"
                                 else entry_price - sl_dist * TP_RR)
                partial_tp_price = (entry_price + sl_dist * PARTIAL_TP_R if direction == "long"
                                    else entry_price - sl_dist * PARTIAL_TP_R)
                trail_sl      = sl_price
                trail_active  = False
                partial_done  = False
                bars_in_trade = 0
                in_trade      = True
            elif pb_bars_left <= 0:
                pb_pending = False   # window expired

        # ── Signal detection + entry (or queue pullback) ──────────────────────
        if not in_trade and not pb_pending:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            if ADX_MIN > 0 and (pd.isna(row["adx"]) or row["adx"] < ADX_MIN):
                continue
            if ADX_SLOPE_BARS > 0 and not row["adx_slope_ok"]:
                continue
            if not row["vol_ok"]:
                continue

            if   row["entry_long"]  and     row["trend_up"]:
                sig = "long"
            elif row["entry_short"] and not row["trend_up"]:
                sig = "short"
            else:
                continue

            if USE_PULLBACK:
                pb_pending   = True
                pb_direction = sig
                pb_level     = row["don_upper"] if sig == "long" else row["don_lower"]
                pb_atr       = atr
                pb_bars_left = PULLBACK_WINDOW
            else:
                direction   = sig
                entry_price = row["close"]
                sl_dist     = atr * SL_MULT
                sl_dist_pct = sl_dist / entry_price
                if USE_VOL_TARGET:
                    rv = row["realised_vol"]
                    notional_full = (capital * VOL_TARGET / rv
                                     if not pd.isna(rv) and rv > 1e-6
                                     else capital * BASE_RISK / sl_dist_pct)
                else:
                    notional_full = capital * BASE_RISK / sl_dist_pct
                notional_full = min(notional_full, capital * LEVERAGE)
                notional      = notional_full
                sl_price      = (entry_price - sl_dist if direction == "long"
                                 else entry_price + sl_dist)
                tp_price      = (entry_price + sl_dist * TP_RR if direction == "long"
                                 else entry_price - sl_dist * TP_RR)
                partial_tp_price = (entry_price + sl_dist * PARTIAL_TP_R if direction == "long"
                                    else entry_price - sl_dist * PARTIAL_TP_R)
                trail_sl      = sl_price
                trail_active  = False
                partial_done  = False
                bars_in_trade = 0
                in_trade      = True

    if not trades:
        print("  No trades.")
        return None

    t = pd.DataFrame(trades)
    t.to_csv(RESULTS_DIR / f"{coin}_calmar.csv", index=False)

    m     = compute_metrics(t, INITIAL_CAPITAL)
    pf_s  = f"{m['pf']:.2f}"     if m["pf"]     < 999 else "inf"
    cal_s = f"{m['calmar']:.2f}" if m["calmar"]  < 999 else "inf"

    size_mode = f"VolTarget({VOL_TARGET*100:.0f}%)" if USE_VOL_TARGET else f"FixedRisk({BASE_RISK*100:.1f}%)"
    print(f"  Sizing        : {size_mode}  |  Max lev {LEVERAGE}x")
    print(f"  Trades        : {m['n']}  ({m['wins']}W / {m['losses']}L,  {m['win_rate']:.1f}%)")
    print(f"  TP / SL       : {(t['exit_reason']=='TP').sum()} / {(t['exit_reason']=='SL').sum()}")
    print(f"  Avg win/loss  : ${m['avg_win']:.2f} / ${m['avg_loss']:.2f}")
    print(f"  Expectancy    : ${m['expectancy']:.2f} / trade")
    print(f"  Max hold ratio: {t['peak_loss_ratio'].max()*100:.1f}%")
    print(f"  Profit factor : {pf_s}")
    print(f"  Total PnL     : ${t['pnl_usdt'].sum():.2f}")
    print(f"  Ann return    : {m['ann_ret']:.1f}%")
    print(f"  Max drawdown  : {m['max_dd']:.1f}%")
    print(f"  Sharpe (ann)  : {m['sharpe']:.2f}")
    print(f"  Calmar        : {cal_s}")
    print(f"  Final capital : ${m['final']:.2f}")
    return t, _score(m), t["peak_loss_ratio"].max(), m


# ── Multi-coin run ────────────────────────────────────────────────────────────
def run_once(verbose: bool = True, coins=None) -> tuple:
    active_coins = coins if coins is not None else COINS
    coin_scores:  dict = {}
    coin_holds:   dict = {}
    coin_metrics: dict = {}
    for symbol, coin in active_coins:
        result = run_backtest(symbol, coin) if verbose else _run_silent(symbol, coin)
        if result is not None:
            _, sc, hold, m = result
            coin_scores[coin]  = sc
            coin_holds[coin]   = hold
            coin_metrics[coin] = m
    avg_score = sum(coin_scores.values()) / len(coin_scores) if coin_scores else float("-inf")
    return avg_score, coin_scores, coin_holds, coin_metrics


def _run_silent(symbol: str, coin: str):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(symbol, coin)


# ── Helpers ───────────────────────────────────────────────────────────────────
def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK,
        "USE_VOL_TARGET": USE_VOL_TARGET, "VOL_TARGET": VOL_TARGET, "VOL_LOOKBACK": VOL_LOOKBACK,
        "DONCHIAN_PERIOD": DONCHIAN_PERIOD, "ATR_PERIOD": ATR_PERIOD,
        "SL_MULT": SL_MULT, "TP_RR": TP_RR, "TREND_EMA_PERIOD": TREND_EMA_PERIOD,
        "ADX_PERIOD": ADX_PERIOD, "ADX_MIN": ADX_MIN, "ADX_SLOPE_BARS": ADX_SLOPE_BARS,
        "VOL_MA_PERIOD": VOL_MA_PERIOD, "VOL_MULT": VOL_MULT,
        "USE_TRAIL": USE_TRAIL, "TRAIL_TRIGGER_R": TRAIL_TRIGGER_R, "TRAIL_MULT": TRAIL_MULT,
        "COOLDOWN_BARS": COOLDOWN_BARS,
        "USE_PARTIAL_TP": USE_PARTIAL_TP, "PARTIAL_TP_R": PARTIAL_TP_R, "PARTIAL_TP_FRAC": PARTIAL_TP_FRAC,
        "USE_PULLBACK": USE_PULLBACK, "PULLBACK_ATR": PULLBACK_ATR, "PULLBACK_WINDOW": PULLBACK_WINDOW,
        "MAX_HOLD_BARS": MAX_HOLD_BARS,
        "OPTIMIZE_TARGET": OPTIMIZE_TARGET,
    }


def _apply_params(p: dict):
    g = globals()
    for k, v in p.items():
        g[k] = v


def _worker_init():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
        )


def _tune_worker(p: dict):
    coins_filter = p.pop('_coins', None)
    _apply_params(p)
    avg_score, coin_scores, coin_holds, _ = run_once(verbose=False, coins=coins_filter)
    return p, avg_score, coin_scores, coin_holds, current_params()


def _save_best_results_table():
    if not BEST_PARAMS_FILE.exists():
        return
    best = json.loads(BEST_PARAMS_FILE.read_text())
    coin_metrics: dict = {}
    for symbol, coin in COINS:
        entry = best.get(coin)
        if not entry:
            continue
        _apply_params(entry["params"])
        result = _run_silent(symbol, coin)
        if result is not None:
            _, _sc, _hold, m = result
            coin_metrics[coin] = m

    if not coin_metrics:
        return

    import io, sys as _sys
    buf = io.StringIO()
    _old, _sys.stdout = _sys.stdout, buf
    try:
        print_summary_table(
            f"Calmar-Optimised Breakout (target={OPTIMIZE_TARGET})",
            f"per-coin optimal  |  {len(coin_metrics)} coins",
            coin_metrics,
        )
    finally:
        _sys.stdout = _old
    table_text = buf.getvalue()

    print(table_text)
    out_file = RESULTS_DIR / "best_results_table.txt"
    out_file.write_text(table_text, encoding="utf-8")
    print(f"Best results table saved to {out_file}")


# ── Auto-tune (multi-process grid search) ─────────────────────────────────────
def auto_tune(coins=None):
    import itertools
    import multiprocessing as mp

    active_coins = coins if coins is not None else COINS
    keys   = list(TUNE_SPACE.keys())
    values = list(TUNE_SPACE.values())
    combos_raw = [{**dict(zip(keys, c)), '_coins': active_coins} for c in itertools.product(*values)]

    # Deduplicate: PARTIAL_TP_R is irrelevant when USE_PARTIAL_TP=False
    seen   = set()
    combos = []
    for p in combos_raw:
        key_parts = {k: v for k, v in p.items() if k != '_coins'}
        if not p.get("USE_PARTIAL_TP"):
            key_parts["PARTIAL_TP_R"] = None
        fingerprint = tuple(sorted(key_parts.items()))
        if fingerprint not in seen:
            seen.add(fingerprint)
            combos.append(p)

    total  = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))

    print(f"\n{'='*65}")
    print(f"  CALMAR AUTO-TUNE  |  target={OPTIMIZE_TARGET}  |  {total} combos  |  {len(active_coins)} coins")
    print(f"  Workers           |  {n_workers} parallel processes (spawn)")
    print(f"{'='*65}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="CALMAR-TUNE", unit="combo", ncols=95)
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for p, avg_score, coin_scores, coin_holds, snapped_params in \
                pool.imap_unordered(_tune_worker, combos, chunksize=8):
            done += 1
            pbar.update(1)

            updated = []
            for coin, sc in coin_scores.items():
                if sc == float("-inf"):
                    continue
                prev_sc = best.get(coin, {}).get("best_score", float("-inf"))
                if sc > prev_sc:
                    best[coin] = {
                        "best_score":      round(sc, 6),
                        "max_hold_ratio":  round(coin_holds.get(coin, 0), 6),
                        "optimize_target": OPTIMIZE_TARGET,
                        "params":          snapped_params,
                    }
                    updated.append(f"{coin.upper()} {OPTIMIZE_TARGET}={sc:.3f}")

            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  ★ {', '.join(updated)}"
                    f"  | lev={p['LEVERAGE']} vt={p['USE_VOL_TARGET']} don={p['DONCHIAN_PERIOD']}"
                    f" sl={p['SL_MULT']} rr={p['TP_RR']} adx={p['ADX_MIN']}"
                )
            elif done % 100 == 0:
                pbar.write(f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  (no improvement)")
    pbar.close()
    print(f"\nTuning complete. Best per-coin results in {BEST_PARAMS_FILE}")
    _save_best_results_table()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--coin', type=str, default=None,
                        help='Run autotune/backtest for a single coin only (e.g. --coin sui)')
    args, _ = parser.parse_known_args()
    coins_filter = None
    if args.coin:
        coins_filter = [(s, c) for s, c in COINS if c == args.coin.lower()]
        if not coins_filter:
            print(f"Unknown coin '{args.coin}'. Available: {[c for _, c in COINS]}")
            return
        print(f"[--coin] Filtering to: {coins_filter}")

    if AUTO_TUNE:
        auto_tune(coins=coins_filter)
        return

    size_mode = f"VolTarget({VOL_TARGET*100:.0f}%)" if USE_VOL_TARGET else f"FixedRisk({BASE_RISK*100:.1f}%)"
    trail_info = f"ON (trig={TRAIL_TRIGGER_R}R, mult={TRAIL_MULT})" if USE_TRAIL else "OFF"
    print("Calmar-Optimised Trend Breakout")
    print(f"Capital ${INITIAL_CAPITAL:,}  |  {size_mode}  |  Max lev {LEVERAGE}x")
    print(f"Donchian({DONCHIAN_PERIOD})  |  ATR({ATR_PERIOD})x{SL_MULT}  |  RR {TP_RR}:1  |  EMA{TREND_EMA_PERIOD}")
    print(f"ADX({ADX_PERIOD})>={ADX_MIN}  |  Vol>={VOL_MULT}x  |  Trail {trail_info}  |  Cooldown {COOLDOWN_BARS}b")

    preload_data()
    avg_score, coin_scores, coin_holds, coin_metrics = run_once(verbose=True)
    print(f"\nAvg {OPTIMIZE_TARGET} across coins: {avg_score:.3f}")

    hdr = f"{size_mode} | Lev {LEVERAGE}x | Don({DONCHIAN_PERIOD}) | RR {TP_RR} | ADX>={ADX_MIN}"
    print_summary_table(f"Calmar-Optimised (target={OPTIMIZE_TARGET})", hdr, coin_metrics)

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}
    for coin, sc in coin_scores.items():
        prev_sc = best.get(coin, {}).get("best_score", float("-inf"))
        tag = ""
        if sc > prev_sc:
            best[coin] = {
                "best_score":      round(sc, 6),
                "max_hold_ratio":  round(coin_holds[coin], 6),
                "optimize_target": OPTIMIZE_TARGET,
                "params":          current_params(),
            }
            tag = f"  * new best (prev {prev_sc:.3f})"
        print(f"  {coin.upper()}: {OPTIMIZE_TARGET}={sc:.3f}  |  hold {coin_holds[coin]*100:.1f}%"
              f"  |  best {best[coin]['best_score']:.3f}{tag}")

    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nLogs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
