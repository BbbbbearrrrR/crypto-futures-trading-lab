"""
5m VWAP Pullback Scalper
========================
Entry   : trade 5m pullback continuation in the direction of the 1h trend.
Long    : 1h trend up, 5m EMA fast above EMA slow, previous 5m bar pulls back
          into VWAP / EMA fast, current bar closes back above both with RSI bias.
Short   : symmetric.
Filters : 5m ADX strength + volume expansion.
SL      : entry +/- ATR(5m) x SL_MULT
TP      : entry +/- ATR(5m) x SL_MULT x TP_RR, optional partial TP / trail

This is closer to the repeatable part of many crypto intraday KOL playbooks:
follow higher-timeframe bias, wait for intraday pullback to VWAP/EMA, then hit
the momentum continuation instead of chasing the first impulse.
"""

# Must be set BEFORE numpy/pandas import to prevent fork deadlock
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
RESULTS_DIR = _ROOT / "results/vwap_5m"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RAW_DATA: dict = {}
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# Parameters
INITIAL_CAPITAL = 10_000
FEE_RATE        = 0.0005
LEVERAGE        = 5

BASE_RISK       = 0.008
ATR_PERIOD      = 14
SL_MULT         = 1.2
TP_RR           = 2.0

EMA_FAST_PERIOD = 12
EMA_SLOW_PERIOD = 34
HTF_EMA_PERIOD  = 200          # 1h EMA for directional bias
RSI_PERIOD      = 14
RSI_BIAS        = 5.0          # long >= 55, short <= 45

USE_VOL_TARGET  = True
VOL_TARGET      = 0.20
VOL_LOOKBACK    = 72           # 72 x 5m = 6h realised vol estimate

ADX_PERIOD      = 14
ADX_MIN         = 15.0
VOL_MA_PERIOD   = 20
VOL_MULT        = 1.2

USE_TRAIL       = False
TRAIL_TRIGGER_R = 1.0
TRAIL_MULT      = 1.0

COOLDOWN_BARS   = 3

USE_PARTIAL_TP  = True
PARTIAL_TP_R    = 1.0
PARTIAL_TP_FRAC = 0.5

MAX_HOLD_BARS   = 24           # 24 x 5m = 2h

AUTO_TUNE       = False        # 5m data is large; verify single-run first
OPTIMIZE_TARGET = "calmar"
MIN_TRADE_COUNT = 80

TUNE_SPACE = {
    "LEVERAGE":         [3, 5],
    "USE_VOL_TARGET":   [True, False],
    "VOL_TARGET":       [0.15, 0.20],
    "EMA_FAST_PERIOD":  [12, 20],
    "EMA_SLOW_PERIOD":  [34, 50],
    "HTF_EMA_PERIOD":   [100, 200],
    "RSI_BIAS":         [3.0, 5.0],
    "ADX_MIN":          [10.0, 20.0],
    "SL_MULT":          [1.0, 1.2, 1.5],
    "TP_RR":            [1.5, 2.0],
    "VOL_MULT":         [1.0, 1.2],
    "USE_PARTIAL_TP":   [True, False],
    "MAX_HOLD_BARS":    [12, 24],
    "COOLDOWN_BARS":    [0, 3],
}

COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
]


def prepare(df_5m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    df = df_5m.copy()

    df["ema_fast"] = df["close"].ewm(span=EMA_FAST_PERIOD, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=EMA_SLOW_PERIOD, adjust=False).mean()

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["realised_vol"] = log_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(12 * 24 * 365)

    up_move  = df["high"] - df["high"].shift(1)
    dn_move  = df["low"].shift(1) - df["low"]
    plus_dm  = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    atr_dx   = tr.ewm(span=ADX_PERIOD, adjust=False).mean().clip(lower=1e-9)
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=ADX_PERIOD, adjust=False).mean() / atr_dx
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9)
    df["adx"] = dx.ewm(span=ADX_PERIOD, adjust=False).mean()

    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs    = gain / loss.clip(lower=1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))

    if "volume" in df.columns:
        df["vol_ma"] = df["volume"].rolling(VOL_MA_PERIOD).mean()
        df["vol_ok"] = df["volume"] >= df["vol_ma"] * VOL_MULT
    else:
        df["vol_ok"] = True

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    session = df.index.floor("D")
    cum_pv  = (typical * df["volume"]).groupby(session).cumsum()
    cum_vol = df["volume"].groupby(session).cumsum().replace(0, np.nan)
    df["vwap"] = cum_pv / cum_vol

    h1 = df_1h.copy()
    h1_ema = h1["close"].ewm(span=HTF_EMA_PERIOD, adjust=False).mean()
    h1["trend_up"] = h1["close"] > h1_ema
    trend = h1["trend_up"].reindex(df.index, method="ffill").ffill()
    df["trend_up"] = np.where(trend.isna(), False, trend).astype(bool)

    df["bar_range"] = (df["high"] - df["low"]).clip(lower=1e-9)
    df["body"] = (df["close"] - df["open"]).abs()
    df["close_pos"] = (df["close"] - df["low"]) / df["bar_range"]
    df["ema_fast_slope_up"] = df["ema_fast"] > df["ema_fast"].shift(3)
    df["ema_fast_slope_dn"] = df["ema_fast"] < df["ema_fast"].shift(3)

    prev_pullback_long = (
        ((df["low"].shift(1) <= df["ema_fast"].shift(1)) |
         (df["low"].shift(1) <= df["vwap"].shift(1))) &
        (df["close"].shift(1) >= df["ema_slow"].shift(1))
    )
    prev_pullback_short = (
        ((df["high"].shift(1) >= df["ema_fast"].shift(1)) |
         (df["high"].shift(1) >= df["vwap"].shift(1))) &
        (df["close"].shift(1) <= df["ema_slow"].shift(1))
    )

    long_reclaim = (
        (df["close"] > df["high"].shift(1)) &
        (df["close"] > df["ema_fast"]) &
        (df["close"] > df["vwap"]) &
        (df["body"] >= 0.25 * df["atr"]) &
        (df["close_pos"] >= 0.65) &
        (((df["close"] - df["ema_fast"]) / df["atr"].clip(lower=1e-9)) <= 1.5)
    )
    short_reclaim = (
        (df["close"] < df["low"].shift(1)) &
        (df["close"] < df["ema_fast"]) &
        (df["close"] < df["vwap"]) &
        (df["body"] >= 0.25 * df["atr"]) &
        (df["close_pos"] <= 0.35) &
        (((df["ema_fast"] - df["close"]) / df["atr"].clip(lower=1e-9)) <= 1.5)
    )

    df["entry_long"] = (
        (df["ema_fast"] > df["ema_slow"]) &
        df["ema_fast_slope_up"] &
        (df["rsi"] >= 50 + RSI_BIAS) &
        prev_pullback_long &
        long_reclaim
    )
    df["entry_short"] = (
        (df["ema_fast"] < df["ema_slow"]) &
        df["ema_fast_slope_dn"] &
        (df["rsi"] <= 50 - RSI_BIAS) &
        prev_pullback_short &
        short_reclaim
    )

    return df


def compute_metrics(t: pd.DataFrame, initial_capital: float) -> dict:
    n        = len(t)
    wins     = int((t["pnl_usdt"] > 0).sum())
    losses   = n - wins
    win_rate = wins / n * 100
    avg_win  = float(t.loc[t["pnl_usdt"] > 0,  "pnl_usdt"].mean()) if wins else 0.0
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
    if m is None or m["n"] < MIN_TRADE_COUNT:
        return float("-inf")
    if OPTIMIZE_TARGET == "calmar":
        v = m["calmar"]
        return v if v < 1e6 else 0.0
    if OPTIMIZE_TARGET == "sharpe":
        return m["sharpe"]
    return m["total_ret"] / 100


def print_summary_table(strategy_name: str, header: str, metrics: dict):
    cols   = ["Coin", "Trades", "Win%", "AvgWin$", "AvgLoss$", "PF",
              "Expect$", "Ann%", "MaxDD%", "Sharpe", "Calmar"]
    widths = [5, 7, 6, 9, 9, 6, 8, 7, 7, 7, 7]

    def _sep(l, m, r):
        return l + m.join("-" * (w + 2) for w in widths) + r

    def _row(vals):
        return "|" + "|".join(f" {str(v):>{w}} " for v, w in zip(vals, widths)) + "|"

    total_w = sum(w + 3 for w in widths) + 1
    title   = f" {strategy_name}  *  {header} "
    print(f"\n+{'-' * (total_w - 2)}+")
    print(f"|{title:<{total_w - 2}}|")
    print(_sep("+", "+", "+"))
    print(_row(cols))
    print(_sep("+", "+", "+"))
    for coin, m in metrics.items():
        pf_s  = f"{m['pf']:.2f}"     if m["pf"] < 999 else "inf"
        cal_s = f"{m['calmar']:.2f}" if m["calmar"] < 999 else "inf"
        print(_row([
            coin.upper(), m["n"],
            f"{m['win_rate']:.1f}%", f"${m['avg_win']:.1f}", f"${m['avg_loss']:.1f}",
            pf_s, f"${m['expectancy']:.1f}",
            f"{m['ann_ret']:.1f}%", f"{m['max_dd']:.1f}%",
            f"{m['sharpe']:.2f}", cal_s,
        ]))
    print(_sep("+", "+", "+"))


def preload_data():
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
        )


def run_backtest(symbol: str, coin: str):
    print(f"\n{'='*50}")
    print(f"  {symbol}")
    print(f"{'='*50}")

    df_5m, df_1h = _RAW_DATA.get(coin) or (
        pd.read_csv(DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True),
        pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
    )
    df = prepare(df_5m, df_1h)

    capital  = float(INITIAL_CAPITAL)
    peak_cap = capital
    trades   = []

    in_trade           = False
    direction          = None
    entry_price        = 0.0
    sl_price           = 0.0
    tp_price           = 0.0
    partial_tp_price   = 0.0
    partial_done       = False
    notional           = 0.0
    notional_full      = 0.0
    peak_loss_ratio    = 0.0
    trail_active       = False
    trail_sl           = 0.0
    cooldown_remaining = 0
    bars_in_trade      = 0

    warmup = max(EMA_SLOW_PERIOD, ATR_PERIOD, RSI_PERIOD, ADX_PERIOD, VOL_MA_PERIOD, VOL_LOOKBACK) + 2

    _iter = tqdm(range(warmup, len(df)), desc=f"{coin.upper()}",
                 unit="bar", file=sys.stdout,
                 disable=not sys.stdout.isatty(), dynamic_ncols=True)

    for i in _iter:
        row = df.iloc[i]
        ts  = df.index[i]

        if capital <= 10:
            break

        if in_trade:
            bars_in_trade += 1

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

            if USE_PARTIAL_TP and not partial_done:
                hit_partial = (row["high"] >= partial_tp_price if direction == "long"
                               else row["low"] <= partial_tp_price)
                if hit_partial:
                    close_notional = notional_full * PARTIAL_TP_FRAC
                    pct_p = ((partial_tp_price - entry_price) / entry_price if direction == "long"
                             else (entry_price - partial_tp_price) / entry_price)
                    pnl_p = close_notional * pct_p - close_notional * FEE_RATE * 2
                    capital += pnl_p
                    peak_cap = max(peak_cap, capital)
                    notional = notional_full * (1.0 - PARTIAL_TP_FRAC)
                    partial_done = True
                    trail_active = True
                    trail_sl = sl_price
                    trades.append({
                        "exit_time": ts,
                        "direction": direction,
                        "entry_price": round(entry_price, 6),
                        "exit_price": round(partial_tp_price, 6),
                        "notional": round(close_notional, 4),
                        "exit_reason": "PARTIAL_TP",
                        "peak_loss_ratio": round(peak_loss_ratio, 6),
                        "pnl_usdt": round(pnl_p, 4),
                        "capital": round(capital, 4),
                        "drawdown": round((peak_cap - capital) / peak_cap, 6),
                    })
                    peak_loss_ratio = 0.0

            hit_tp   = (row["high"] >= tp_price if direction == "long" else row["low"] <= tp_price)
            hit_sl   = (row["low"] <= sl_price if direction == "long" else row["high"] >= sl_price)
            hit_time = (MAX_HOLD_BARS > 0 and bars_in_trade >= MAX_HOLD_BARS)

            worst_pnl = ((row["low"] - entry_price) / entry_price * notional if direction == "long"
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
                capital += pnl
                peak_cap = max(peak_cap, capital)
                trades.append({
                    "exit_time": ts,
                    "direction": direction,
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(exit_price, 6),
                    "notional": round(notional, 4),
                    "exit_reason": exit_reason,
                    "peak_loss_ratio": round(peak_loss_ratio, 6),
                    "pnl_usdt": round(pnl, 4),
                    "capital": round(capital, 4),
                    "drawdown": round((peak_cap - capital) / peak_cap, 6),
                })
                peak_loss_ratio    = 0.0
                trail_active       = False
                trail_sl           = 0.0
                partial_done       = False
                bars_in_trade      = 0
                in_trade           = False
                if exit_reason == "SL":
                    cooldown_remaining = COOLDOWN_BARS

        if not in_trade:
            if cooldown_remaining > 0:
                cooldown_remaining -= 1
                continue

            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue
            if ADX_MIN > 0 and (pd.isna(row["adx"]) or row["adx"] < ADX_MIN):
                continue
            if not row["vol_ok"]:
                continue

            if row["entry_long"] and row["trend_up"]:
                sig = "long"
            elif row["entry_short"] and not row["trend_up"]:
                sig = "short"
            else:
                continue

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
            sl_price      = (entry_price - sl_dist if direction == "long" else entry_price + sl_dist)
            tp_price      = (entry_price + sl_dist * TP_RR if direction == "long" else entry_price - sl_dist * TP_RR)
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
    t.to_csv(RESULTS_DIR / f"{coin}_vwap_5m.csv", index=False)

    m     = compute_metrics(t, INITIAL_CAPITAL)
    pf_s  = f"{m['pf']:.2f}"     if m["pf"] < 999 else "inf"
    cal_s = f"{m['calmar']:.2f}" if m["calmar"] < 999 else "inf"

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


def run_once(verbose: bool = True) -> tuple:
    coin_scores: dict = {}
    coin_holds: dict = {}
    coin_metrics: dict = {}
    for symbol, coin in COINS:
        result = run_backtest(symbol, coin) if verbose else _run_silent(symbol, coin)
        if result is not None:
            _, sc, hold, m = result
            coin_scores[coin]  = sc
            coin_holds[coin]   = hold
            coin_metrics[coin] = m
    avg_score = sum(coin_scores.values()) / len(coin_scores) if coin_scores else float("-inf")
    return avg_score, coin_scores, coin_holds, coin_metrics


def _run_silent(symbol: str, coin: str):
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(symbol, coin)


def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK,
        "USE_VOL_TARGET": USE_VOL_TARGET, "VOL_TARGET": VOL_TARGET, "VOL_LOOKBACK": VOL_LOOKBACK,
        "ATR_PERIOD": ATR_PERIOD, "SL_MULT": SL_MULT, "TP_RR": TP_RR,
        "EMA_FAST_PERIOD": EMA_FAST_PERIOD, "EMA_SLOW_PERIOD": EMA_SLOW_PERIOD,
        "HTF_EMA_PERIOD": HTF_EMA_PERIOD,
        "RSI_PERIOD": RSI_PERIOD, "RSI_BIAS": RSI_BIAS,
        "ADX_PERIOD": ADX_PERIOD, "ADX_MIN": ADX_MIN,
        "VOL_MA_PERIOD": VOL_MA_PERIOD, "VOL_MULT": VOL_MULT,
        "USE_TRAIL": USE_TRAIL, "TRAIL_TRIGGER_R": TRAIL_TRIGGER_R, "TRAIL_MULT": TRAIL_MULT,
        "COOLDOWN_BARS": COOLDOWN_BARS,
        "USE_PARTIAL_TP": USE_PARTIAL_TP, "PARTIAL_TP_R": PARTIAL_TP_R, "PARTIAL_TP_FRAC": PARTIAL_TP_FRAC,
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
            pd.read_csv(DATA_DIR / f"{coin}_futures_5m.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
        )


def _tune_worker(p: dict):
    _apply_params(p)
    avg_score, coin_scores, coin_holds, _ = run_once(verbose=False)
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

    import io
    import sys as _sys

    buf = io.StringIO()
    _old, _sys.stdout = _sys.stdout, buf
    try:
        print_summary_table(
            f"5m VWAP Pullback (target={OPTIMIZE_TARGET})",
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


def auto_tune():
    import itertools
    import multiprocessing as mp

    keys   = list(TUNE_SPACE.keys())
    values = list(TUNE_SPACE.values())
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]

    total = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))

    print(f"\n{'='*65}")
    print(f"  5M VWAP AUTO-TUNE  |  target={OPTIMIZE_TARGET}  |  {total} combos  |  {len(COINS)} coins")
    print(f"  Workers            |  {n_workers} parallel processes (spawn)")
    print(f"{'='*65}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="VWAP-5M-TUNE", unit="combo", ncols=95)
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
                        "best_score": round(sc, 6),
                        "max_hold_ratio": round(coin_holds.get(coin, 0), 6),
                        "optimize_target": OPTIMIZE_TARGET,
                        "params": snapped_params,
                    }
                    updated.append(f"{coin.upper()} {OPTIMIZE_TARGET}={sc:.3f}")

            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  * {', '.join(updated)}"
                    f"  | lev={p['LEVERAGE']} vf={p['USE_VOL_TARGET']} ema={p['EMA_FAST_PERIOD']}/{p['EMA_SLOW_PERIOD']}"
                    f" sl={p['SL_MULT']} rr={p['TP_RR']} adx={p['ADX_MIN']}"
                )
            elif done % 100 == 0:
                pbar.write(f"  [{done:>{len(str(total))}}/{total}]  avg {avg_score:.3f}  (no improvement)")
    pbar.close()
    print(f"\nTuning complete. Best per-coin results in {BEST_PARAMS_FILE}")
    _save_best_results_table()


def main():
    if AUTO_TUNE:
        auto_tune()
        return

    size_mode = f"VolTarget({VOL_TARGET*100:.0f}%)" if USE_VOL_TARGET else f"FixedRisk({BASE_RISK*100:.1f}%)"
    trail_info = f"ON (trig={TRAIL_TRIGGER_R}R, mult={TRAIL_MULT})" if USE_TRAIL else "OFF"
    print("5m VWAP Pullback Scalper")
    print(f"Capital ${INITIAL_CAPITAL:,}  |  {size_mode}  |  Max lev {LEVERAGE}x")
    print(f"EMA {EMA_FAST_PERIOD}/{EMA_SLOW_PERIOD}  |  1h EMA {HTF_EMA_PERIOD}  |  RSI bias {RSI_BIAS:.1f}")
    print(f"ATR({ATR_PERIOD})x{SL_MULT}  |  RR {TP_RR}:1  |  ADX({ADX_PERIOD})>={ADX_MIN}  |  Vol>={VOL_MULT}x")
    print(f"PartialTP {USE_PARTIAL_TP}  |  MaxHold {MAX_HOLD_BARS} bars  |  Trail {trail_info}  |  Cooldown {COOLDOWN_BARS}b")

    preload_data()
    avg_score, coin_scores, coin_holds, coin_metrics = run_once(verbose=True)
    print(f"\nAvg {OPTIMIZE_TARGET} across coins: {avg_score:.3f}")

    hdr = (
        f"{size_mode} | EMA {EMA_FAST_PERIOD}/{EMA_SLOW_PERIOD} | 1hEMA {HTF_EMA_PERIOD} "
        f"| RR {TP_RR} | ADX>={ADX_MIN}"
    )
    print_summary_table(f"5m VWAP Pullback (target={OPTIMIZE_TARGET})", hdr, coin_metrics)

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}
    for coin, sc in coin_scores.items():
        prev_sc = best.get(coin, {}).get("best_score", float("-inf"))
        tag = ""
        if sc > prev_sc:
            best[coin] = {
                "best_score": round(sc, 6),
                "max_hold_ratio": round(coin_holds[coin], 6),
                "optimize_target": OPTIMIZE_TARGET,
                "params": current_params(),
            }
            tag = f"  * new best (prev {prev_sc:.3f})"
        print(f"  {coin.upper()}: {OPTIMIZE_TARGET}={sc:.3f}  |  hold {coin_holds[coin]*100:.1f}%"
              f"  |  best {best[coin]['best_score']:.3f}{tag}")

    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nLogs -> {RESULTS_DIR}/")


if __name__ == "__main__":
    main()