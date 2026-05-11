"""
Trend Breakout Backtest
=======================
Entry  : 1h close breaks above Donchian upper (N-bar high) → long  (only when 1d trend up)
         1h close breaks below Donchian lower (N-bar low)  → short (only when 1d trend down)
Trend  : 1d close > EMA(TREND_EMA_PERIOD) → trend_up
SL     : entry ± ATR(ATR_PERIOD) × SL_MULT
TP     : entry ± ATR(ATR_PERIOD) × SL_MULT × TP_RR
Size   : risk_amount = capital × BASE_RISK
         notional = risk_amount / sl_distance_pct, capped at capital × LEVERAGE
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

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results/breakout")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_RAW_DATA: dict = {}
BEST_PARAMS_FILE = RESULTS_DIR / "best_params.json"

# ── Parameters ────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000
FEE_RATE         = 0.0005      # 0.05% per side (taker)
LEVERAGE         = 10

BASE_RISK        = 0.01        # fraction of capital risked per trade
DONCHIAN_PERIOD  = 20          # N bars for Donchian channel high/low
ATR_PERIOD       = 14          # ATR smoothing period
SL_MULT          = 1.5         # SL distance = ATR × SL_MULT
TP_RR            = 3.0         # TP distance = SL distance × TP_RR
TREND_EMA_PERIOD = 200         # 1d EMA for trend filter

# ── Auto-tuning ───────────────────────────────────────────────────────────────
AUTO_TUNE = True               # True = grid search; False = single run with above params

TUNE_SPACE = {
    "LEVERAGE":          [10, 20],
    "BASE_RISK":         [0.01, 0.02],
    "DONCHIAN_PERIOD":   [10, 20, 40],
    "ATR_PERIOD":        [7, 14],
    "SL_MULT":           [1.0, 1.5, 2.0],
    "TP_RR":             [2.0, 3.0, 5.0],
    "TREND_EMA_PERIOD":  [50, 100, 200],
}
# Total: 2×2×3×2×3×3×3 = 648 combinations


COINS = [
    ("BTC/USDT:USDT", "btc"),
    ("ETH/USDT:USDT", "eth"),
    ("SOL/USDT:USDT", "sol"),
    ("HYPE/USDT:USDT", "hype"),
]


# ── Indicators ────────────────────────────────────────────────────────────────
def prepare(df_1h: pd.DataFrame, df_1d: pd.DataFrame) -> pd.DataFrame:
    df = df_1h.copy()

    # Donchian channel: shift(1) to avoid look-ahead bias
    df["don_upper"] = df["high"].rolling(DONCHIAN_PERIOD).max().shift(1)
    df["don_lower"] = df["low"].rolling(DONCHIAN_PERIOD).min().shift(1)

    # ATR (Wilder's smoothed)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()

    # Breakout entry signals
    df["entry_long"]  = df["close"] > df["don_upper"]
    df["entry_short"] = df["close"] < df["don_lower"]

    # 1d trend filter: close > EMA(TREND_EMA_PERIOD) → trend_up
    d1     = df_1d.copy()
    d1_ema = d1["close"].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()
    d1["trend_up"] = d1["close"] > d1_ema
    trend = d1["trend_up"].reindex(df.index, method="ffill")
    _t    = trend.ffill()
    df["trend_up"] = np.where(_t.isna(), False, _t).astype(bool)

    return df


# ── Backtest ──────────────────────────────────────────────────────────────────
def preload_data():
    """Load all raw CSV files into _RAW_DATA once per worker process."""
    global _RAW_DATA
    for _, coin in COINS:
        _RAW_DATA[coin] = (
            pd.read_csv(DATA_DIR / f"{coin}_futures_1h.csv", index_col=0, parse_dates=True),
            pd.read_csv(DATA_DIR / f"{coin}_futures_1d.csv", index_col=0, parse_dates=True),
        )


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

    in_trade       = False
    direction      = None
    entry_price    = 0.0
    sl_price       = 0.0
    tp_price       = 0.0
    notional       = 0.0
    peak_loss_ratio = 0.0

    warmup = max(DONCHIAN_PERIOD, ATR_PERIOD) + 2

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
            hit_tp = (row["high"] >= tp_price if direction == "long" else row["low"]  <= tp_price)
            hit_sl = (row["low"]  <= sl_price if direction == "long" else row["high"] >= sl_price)

            # track peak unrealized loss ratio this bar
            worst_pnl = ((row["low"]  - entry_price) / entry_price * notional if direction == "long"
                         else (entry_price - row["high"]) / entry_price * notional)
            if worst_pnl < 0:
                peak_loss_ratio = max(peak_loss_ratio, -worst_pnl / capital)

            if hit_tp or hit_sl:
                exit_price = tp_price if hit_tp else sl_price
                pct        = ((exit_price - entry_price) / entry_price if direction == "long"
                              else (entry_price - exit_price) / entry_price)
                pnl        = notional * pct - notional * FEE_RATE * 2
                pnl        = max(pnl, -capital)
                capital   += pnl
                peak_cap   = max(peak_cap, capital)
                trades.append({
                    "exit_time":      ts,
                    "direction":      direction,
                    "entry_price":    round(entry_price, 6),
                    "exit_price":     round(exit_price, 6),
                    "notional":       round(notional, 4),
                    "exit_reason":    "TP" if hit_tp else "SL",
                    "peak_loss_ratio": round(peak_loss_ratio, 6),
                    "pnl_usdt":       round(pnl, 4),
                    "capital":        round(capital, 4),
                    "drawdown":       round((peak_cap - capital) / peak_cap, 6),
                })
                peak_loss_ratio = 0.0
                in_trade = False

        # ── Entry ─────────────────────────────────────────────────────────────
        if not in_trade:
            atr = row["atr"]
            if pd.isna(atr) or atr <= 0:
                continue

            if   row["entry_long"]  and     row["trend_up"]:
                direction = "long"
            elif row["entry_short"] and not row["trend_up"]:
                direction = "short"
            else:
                continue

            entry_price  = row["close"]
            sl_dist      = atr * SL_MULT
            sl_dist_pct  = sl_dist / entry_price
            notional     = min(capital * BASE_RISK / sl_dist_pct, capital * LEVERAGE)
            sl_price     = entry_price - sl_dist if direction == "long" else entry_price + sl_dist
            tp_price     = entry_price + sl_dist * TP_RR if direction == "long" else entry_price - sl_dist * TP_RR
            in_trade     = True

    if not trades:
        print("  No trades.")
        return None

    t = pd.DataFrame(trades)
    t.to_csv(RESULTS_DIR / f"{coin}_breakout.csv", index=False)

    n        = len(t)
    wins     = (t["pnl_usdt"] > 0).sum()
    losses   = n - wins
    win_rate = wins / n * 100
    avg_win  = t.loc[t["pnl_usdt"] > 0,  "pnl_usdt"].mean() if wins   else 0
    avg_loss = t.loc[t["pnl_usdt"] <= 0, "pnl_usdt"].mean() if losses else 0
    pf       = (wins * avg_win / (-losses * avg_loss)
                if losses and avg_loss else float("inf"))
    final    = t["capital"].iloc[-1]

    max_hold_ratio = t["peak_loss_ratio"].max()

    print(f"  Trades        : {n}  ({wins}W / {losses}L,  {win_rate:.1f}%)")
    print(f"  TP / SL       : {(t['exit_reason']=='TP').sum()} / {(t['exit_reason']=='SL').sum()}")
    print(f"  Avg win/loss  : ${avg_win:.2f} / ${avg_loss:.2f}")
    print(f"  Max hold ratio : {max_hold_ratio*100:.1f}%  (peak unrealized loss / capital)")
    print(f"  Profit factor : {pf:.2f}")
    print(f"  Total PnL     : ${t['pnl_usdt'].sum():.2f}")
    print(f"  Total return  : {(final - INITIAL_CAPITAL)/INITIAL_CAPITAL*100:.1f}%")
    print(f"  Max drawdown  : {t['drawdown'].max()*100:.1f}%")
    print(f"  Final capital : ${final:.2f}")
    return t, (final - INITIAL_CAPITAL) / INITIAL_CAPITAL, max_hold_ratio


# ── Helpers ───────────────────────────────────────────────────────────────────
def current_params() -> dict:
    return {
        "LEVERAGE": LEVERAGE, "BASE_RISK": BASE_RISK,
        "DONCHIAN_PERIOD": DONCHIAN_PERIOD, "ATR_PERIOD": ATR_PERIOD,
        "SL_MULT": SL_MULT, "TP_RR": TP_RR,
        "TREND_EMA_PERIOD": TREND_EMA_PERIOD,
    }


def run_once(verbose: bool = True) -> tuple[float, dict, dict]:
    coin_returns: dict = {}
    coin_hold_ratios: dict = {}
    for symbol, coin in COINS:
        result = run_backtest(symbol, coin) if verbose else _run_silent(symbol, coin)
        if result is not None:
            _, ret, hold_ratio = result
            coin_returns[coin] = ret
            coin_hold_ratios[coin] = hold_ratio
    avg_ret = sum(coin_returns.values()) / len(coin_returns) if coin_returns else 0.0
    return avg_ret, coin_returns, coin_hold_ratios


def _run_silent(symbol: str, coin: str):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return run_backtest(symbol, coin)


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
    _apply_params(p)
    avg_ret, coin_returns, coin_hold_ratios = run_once(verbose=False)
    return p, avg_ret, coin_returns, coin_hold_ratios, current_params()


def auto_tune():
    import itertools
    import multiprocessing as mp
    keys   = list(TUNE_SPACE.keys())
    values = list(TUNE_SPACE.values())
    combos = [dict(zip(keys, c)) for c in itertools.product(*values)]
    total  = len(combos)
    n_workers = min(16, max(1, mp.cpu_count() - 1))
    print(f"\n{'='*60}")
    print(f"  AUTO-TUNE  |  {total} combinations  |  {len(COINS)} coins each")
    print(f"  Workers    |  {n_workers} parallel processes (spawn)")
    print(f"{'='*60}")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    ctx  = mp.get_context("spawn")
    done = 0
    pbar = tqdm(total=total, desc="AUTO-TUNE", unit="combo", ncols=90)
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for p, avg_ret, coin_returns, coin_hold_ratios, snapped_params in \
                pool.imap_unordered(_tune_worker, combos, chunksize=1):
            done += 1
            pbar.update(1)

            updated = []
            for coin, ret in coin_returns.items():
                prev_ret = best.get(coin, {}).get("best_return", float("-inf"))
                if ret > prev_ret:
                    best[coin] = {
                        "best_return": round(ret, 6),
                        "max_hold_ratio": round(coin_hold_ratios[coin], 6),
                        "params": snapped_params,
                    }
                    updated.append(f"{coin.upper()} {ret*100:.1f}%")

            if updated:
                BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
                pbar.write(
                    f"  [{done:>{len(str(total))}}/{total}]  avg {avg_ret*100:.1f}%  ★ {', '.join(updated)}"
                    f"  | lev={p['LEVERAGE']} risk={p['BASE_RISK']}"
                    f" don={p['DONCHIAN_PERIOD']} atr={p['ATR_PERIOD']}"
                    f" sl={p['SL_MULT']} rr={p['TP_RR']} ema={p['TREND_EMA_PERIOD']}"
                )
            elif done % 50 == 0:
                pbar.write(f"  [{done:>{len(str(total))}}/{total}]  avg {avg_ret*100:.1f}%  (no improvement)")
    pbar.close()
    print(f"\nTuning complete. Best per-coin results in {BEST_PARAMS_FILE}")


def main():
    if AUTO_TUNE:
        auto_tune()
        return

    print("Trend Breakout Backtest")
    print(f"Capital ${INITIAL_CAPITAL:,}  |  Base risk {BASE_RISK*100:.1f}%/trade  "
          f"|  Leverage {LEVERAGE}×")
    print(f"Donchian({DONCHIAN_PERIOD})  |  ATR({ATR_PERIOD}) × {SL_MULT} SL  "
          f"|  RR {TP_RR}:1  |  1d EMA{TREND_EMA_PERIOD} trend filter")

    avg_return, coin_returns, coin_hold_ratios = run_once(verbose=True)
    print(f"\nAvg return across coins: {avg_return*100:.1f}%")

    best: dict = json.loads(BEST_PARAMS_FILE.read_text()) if BEST_PARAMS_FILE.exists() else {}

    for coin, ret in coin_returns.items():
        hold_ratio = coin_hold_ratios[coin]
        prev_ret = best.get(coin, {}).get("best_return", float("-inf"))
        tag = ""
        if ret > prev_ret:
            best[coin] = {
                "best_return": round(ret, 6),
                "max_hold_ratio": round(hold_ratio, 6),
                "params": current_params(),
            }
            tag = f"  ★ new best (prev {prev_ret*100:.1f}%)"
        print(f"  {coin.upper()}: return {ret*100:.1f}%  |  hold ratio {hold_ratio*100:.1f}%"
              f"  |  best {best[coin]['best_return']*100:.1f}%{tag}")

    BEST_PARAMS_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nLogs → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
