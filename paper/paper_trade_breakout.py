#!/usr/bin/env python3
"""
Paper Trading Engine — Breakout Strategy  (v2 – ExitManager refactor)
=======================================================================
Runs the EXACT same signal logic as backtest_breakout.py against live 1h candles.

State is persisted to paper_state_breakout.json between runs.
All fills are appended to paper_trades_breakout.csv.

Usage:
    python paper_trade_breakout.py           # normal start
    python paper_trade_breakout.py --reset   # wipe state and start fresh
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import json, sys, time, traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import ccxt
import pandas as pd
import numpy as np

from backtest import backtest_breakout as bb
from backtest.exit_manager import ExitManager

# ---- Config ------------------------------------------------------------------
USE_TESTNET      = True
INITIAL_CAPITAL  = 10_000.0
WARMUP_1H        = 300
WARMUP_1D        = 500
SLEEP_BUFFER_SEC = 15

STATE_FILE       = _HERE / "paper_state_breakout.json"
TRADE_LOG_FILE   = _HERE / "paper_trades_breakout.csv"
BEST_PARAMS_FILE = _ROOT / "results/breakout/best_params.json"

API_KEY    = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_API_SECRET", "")

COINS = bb.COINS


# ---- Exchange ----------------------------------------------------------------
def make_exchange() -> ccxt.binance:
    ex = ccxt.binance({
        "apiKey":  API_KEY,
        "secret":  API_SECRET,
        "options": {"defaultType": "future"},
    })
    if USE_TESTNET:
        ex.set_sandbox_mode(True)
    return ex

_ex_pub = ccxt.binanceusdm({"enableRateLimit": True})


def fetch_ohlcv(ex, symbol: str, tf: str, limit: int) -> pd.DataFrame:
    raw = _ex_pub.fetch_ohlcv(symbol, tf, limit=limit + 1)
    df  = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "datetime"
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    now_hour = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index < now_hour]
    return df.tail(limit)


# ---- State -------------------------------------------------------------------
def _default_state() -> dict:
    return {
        "capital":            INITIAL_CAPITAL,
        "peak_cap":           INITIAL_CAPITAL,
        "in_trade":           False,
        "direction":          None,
        "entry_price":        0.0,
        "sl_price":           0.0,
        "tp_price":           0.0,
        "notional":           0.0,
        "notional_full":      0.0,
        # ExitManager-serialised state
        "em_sl_dist":         0.0,
        "em_partial_done":    False,
        "em_trail_active":    False,
        "em_trail_ref":       0.0,
        "em_bars_held":       0,
        # Misc
        "cooldown_remaining": 0,
        "open_time":          None,
        "last_processed_ts":  None,
        "trades":             [],
    }


def load_state() -> dict:
    if STATE_FILE.exists():
        data = json.loads(STATE_FILE.read_text())
        for _, coin in COINS:
            if coin not in data:
                data[coin] = _default_state()
            else:
                for k, v in _default_state().items():
                    data[coin].setdefault(k, v)
        return data
    return {coin: _default_state() for _, coin in COINS}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


def _log_trade(rec: dict):
    df  = pd.DataFrame([rec])
    hdr = not TRADE_LOG_FILE.exists()
    df.to_csv(TRADE_LOG_FILE, mode="a", index=False, header=hdr)


def _exit_params_from(params: dict) -> dict:
    """Build ExitManager params dict from best_params entry."""
    return {
        "USE_PARTIAL_TP":  params.get("USE_PARTIAL_TP",  True),
        "PARTIAL_R":       params.get("PARTIAL_R",        1.0),
        "PARTIAL_FRAC":    params.get("PARTIAL_FRAC",     0.5),
        "USE_TRAIL":       params.get("USE_TRAIL",        True),
        "TRAIL_ATR_MULT":  params.get("TRAIL_ATR_MULT",   1.0),
        "TRAIL_TRIGGER_R": params.get("TRAIL_TRIGGER_R",  1.0),
        "TP_RR":           params.get("TP_RR",            3.0),
        "MAX_HOLD_BARS":   params.get("MAX_HOLD_BARS",    72),
        "USE_TREND_EXIT":  params.get("USE_TREND_EXIT",   True),
        "FEE_RATE":        bb.FEE_RATE,
    }


def _restore_em(cs: dict, params: dict) -> ExitManager:
    """Reconstruct ExitManager from serialised coin-state."""
    ep  = cs["entry_price"]
    sl  = cs["sl_price"]
    tp  = cs["tp_price"]
    em  = ExitManager(cs["direction"], ep, sl, tp, _exit_params_from(params))
    # Restore mutable fields
    em.sl_dist     = cs["em_sl_dist"] or abs(ep - sl)
    em.partial_done = cs["em_partial_done"]
    em.trail_active = cs["em_trail_active"]
    em._trail_ref   = cs["em_trail_ref"] or sl
    em.bars_held    = cs["em_bars_held"]
    # Re-sync sl to latest (trail may have moved it)
    em.sl_price     = sl
    return em


def _save_em(cs: dict, em: ExitManager):
    """Persist ExitManager mutable state into coin-state dict."""
    cs["sl_price"]         = em.sl_price
    cs["tp_price"]         = em.tp_price
    cs["em_sl_dist"]       = em.sl_dist
    cs["em_partial_done"]  = em.partial_done
    cs["em_trail_active"]  = em.trail_active
    cs["em_trail_ref"]     = em._trail_ref
    cs["em_bars_held"]     = em.bars_held


# ---- Per-bar processing ------------------------------------------------------
def process_bar(cs: dict, row: pd.Series, params: dict, coin: str, ts) -> list:
    """Process one just-closed candle. Mutates cs. Returns fill records."""
    records  = []
    cap      = cs["capital"]
    peak_cap = cs["peak_cap"]

    # -- Exit ------------------------------------------------------------------
    if cs["in_trade"]:
        em = _restore_em(cs, params)
        result = em.update(row)
        _save_em(cs, em)   # persist updated SL/trail state

        if result.partial:
            pt    = result.partial
            cn    = cs["notional_full"] * pt.frac
            pnl_p = cn * pt.pnl_frac - cn * bb.FEE_RATE * 2
            cap  += pnl_p
            peak_cap = max(peak_cap, cap)
            cs["notional"] = cs["notional_full"] * (1.0 - pt.frac)
            rec = dict(timestamp=str(ts), coin=coin, direction=cs["direction"],
                       entry_price=cs["entry_price"], exit_price=round(pt.exit_price, 6),
                       notional=round(cn, 4), exit_reason=pt.reason,
                       pnl_usdt=round(pnl_p, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            print(f"  [{coin.upper()}] ◑ PARTIAL_TP {cs['direction'].upper()}"
                  f"  exit={pt.exit_price:.4f}  pnl=${pnl_p:+.2f}  cap=${cap:.0f}"
                  f"  @{str(ts)[:16]}")

        if result.closed:
            nt  = cs["notional"]
            pnl = max(nt * result.pnl_frac - nt * bb.FEE_RATE * 2, -cap)
            cap += pnl
            peak_cap = max(peak_cap, cap)
            rec = dict(timestamp=str(ts), coin=coin, direction=cs["direction"],
                       entry_price=cs["entry_price"], exit_price=round(result.exit_price, 6),
                       notional=round(nt, 4), exit_reason=result.reason,
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            records.append(rec)
            _log_trade(rec)
            cs["trades"].append(rec)
            sym = "✓" if pnl >= 0 else "✗"
            print(f"  [{coin.upper()}] {sym} EXIT {cs['direction'].upper()}"
                  f" [{result.reason}]  exit={result.exit_price:.4f}"
                  f"  pnl=${pnl:+.2f}  cap=${cap:.0f}  @{str(ts)[:16]}")
            cs.update({"in_trade": False, "open_time": None})
            if result.reason == "SL":
                cs["cooldown_remaining"] = params.get("COOLDOWN_BARS", 0)

        cs["capital"]  = cap
        cs["peak_cap"] = peak_cap

    # -- Entry -----------------------------------------------------------------
    if not cs["in_trade"]:
        if cs["cooldown_remaining"] > 0:
            cs["cooldown_remaining"] -= 1
            return records

        atr = float(row.get("atr", float("nan")))
        if np.isnan(atr) or atr <= 0:
            return records

        adx_min = params.get("ADX_MIN", 0.0)
        if adx_min > 0:
            adx = float(row.get("adx", float("nan")))
            if np.isnan(adx) or adx < adx_min:
                return records

        if not bool(row.get("vol_ok", True)):
            return records

        trend_up    = bool(row.get("trend_up", False))
        entry_long  = bool(row.get("entry_long", False))
        entry_short = bool(row.get("entry_short", False))

        if   entry_long  and     trend_up: direction = "long"
        elif entry_short and not trend_up: direction = "short"
        else: return records

        # OBV divergence filter
        if params.get("USE_OBV_FILTER", False):
            if direction == "long"  and not bool(row.get("obv_above_ma", True)):
                return records
            if direction == "short" and not bool(row.get("obv_below_ma", True)):
                return records

        ep = float(row["close"])
        sl_mode = params.get("SL_MODE", "donchian")
        if sl_mode == "donchian":
            sl_price = float(row["don_lower"]) if direction == "long" else float(row["don_upper"])
            if direction == "long"  and sl_price >= ep: sl_price = ep - atr * 1.5
            if direction == "short" and sl_price <= ep: sl_price = ep + atr * 1.5
        else:
            sl_mult  = params.get("SL_MULT", 1.5)
            sl_price = ep - atr * sl_mult if direction == "long" else ep + atr * sl_mult

        sl_dist_pct = abs(ep - sl_price) / ep
        if sl_dist_pct < 1e-6:
            return records

        sl_dist  = abs(ep - sl_price)
        tp_rr    = params.get("TP_RR", 3.0)
        tp_price = ep + sl_dist * tp_rr if direction == "long" else ep - sl_dist * tp_rr

        leverage  = params.get("LEVERAGE", 10)
        base_risk = params.get("BASE_RISK", 0.01)
        notional  = min(cap * base_risk / sl_dist_pct, cap * leverage)

        em = ExitManager(direction, ep, sl_price, tp_price, _exit_params_from(params))

        cs.update({
            "in_trade":      True,
            "direction":     direction,
            "entry_price":   ep,
            "sl_price":      sl_price,
            "tp_price":      tp_price,
            "notional":      notional,
            "notional_full": notional,
            "open_time":     str(ts),
        })
        _save_em(cs, em)
        print(f"  [{coin.upper()}] ▶ ENTRY {direction.upper()}"
              f"  price={ep:.4f}  SL={sl_price:.4f}  TP={tp_price:.4f}"
              f"  notional=${notional:.0f}  @{str(ts)[:16]}")

    return records


# ---- Report ------------------------------------------------------------------
def print_report(state: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'═'*65}")
    print(f"  PAPER PORTFOLIO (Breakout v2)  |  {now}")
    print(f"{'═'*65}")
    total_cap = 0.0
    for _, coin in COINS:
        cs   = state[coin]
        cap  = cs["capital"]
        ret  = (cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        total_cap += cap
        n    = len(cs["trades"])
        wins = sum(1 for t in cs["trades"] if t["pnl_usdt"] > 0)
        wr   = f"{wins/n*100:.0f}%" if n else " - "
        pos  = (f"{cs['direction'].upper()} @ {cs['entry_price']:.4f}"
                if cs["in_trade"] else "flat")
        print(f"  {coin.upper():5s}  cap=${cap:>9.2f}  ret={ret:>+7.2f}%"
              f"  trades={n:>3d}  wr={wr:>4s}  [{pos}]")
    total_ret = (total_cap - INITIAL_CAPITAL * len(COINS)) / (INITIAL_CAPITAL * len(COINS)) * 100
    print(f"{'─'*65}")
    print(f"  TOTAL            ${total_cap:>9.2f}  ret={total_ret:>+7.2f}%")
    print(f"{'═'*65}\n")


# ---- Main cycle --------------------------------------------------------------
def run_cycle(ex, state: dict, best: dict):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'─'*65}")
    print(f"  CYCLE  {now}")
    print(f"{'─'*65}")

    for symbol, coin in COINS:
        entry  = best.get(coin, {})
        params = entry.get("params", {})
        if not params:
            print(f"  [{coin.upper()}] no best params, skipping")
            continue

        try:
            bb._apply_params(params)
            df_1h = fetch_ohlcv(ex, symbol, "1h", WARMUP_1H)
            df_1d = fetch_ohlcv(ex, symbol, "1d", WARMUP_1D)
            df    = bb.prepare(df_1h, df_1d)
            cs    = state[coin]

            last_ts = cs.get("last_processed_ts")
            if last_ts:
                missed = df[df.index > pd.Timestamp(last_ts, tz="UTC")]
                if len(missed) == 0:
                    missed = df.iloc[-1:]
            else:
                missed = df.iloc[-1:]

            if len(missed) > 1:
                print(f"  [{coin.upper()}] replaying {len(missed)} missed candles "
                      f"from {str(missed.index[0])[:16]}")

            for ts, row in missed.iterrows():
                adx_str = f"{row.get('adx', float('nan')):.1f}"
                trend   = "↑" if bool(row.get("trend_up", False)) else "↓"
                vol_ok  = "V✓" if bool(row.get("vol_ok", True)) else "V✗"
                el = "L✓" if bool(row.get("entry_long", False))  else "  "
                es = "S✓" if bool(row.get("entry_short", False)) else "  "
                print(f"  [{coin.upper()}]  close={row['close']:.4f}"
                      f"  atr={row.get('atr', 0):.4f}  adx={adx_str}"
                      f"  trend={trend}  {vol_ok}  {el} {es}")
                process_bar(cs, row, params, coin, ts)
                cs["last_processed_ts"] = str(ts)

        except ccxt.NetworkError as e:
            print(f"  [{coin.upper()}] network error: {e}")
        except Exception:
            print(f"  [{coin.upper()}] unexpected error:")
            traceback.print_exc()

    save_state(state)
    print_report(state)


# ---- Intrabar SL monitor -----------------------------------------------------
INTRABAR_CHECK_INTERVAL = 60

def check_intrabar_sl(ex, state: dict, best: dict):
    """Check live price vs SL/TP for open positions between candle closes."""
    changed = False
    for symbol, coin in COINS:
        cs = state[coin]
        if not cs.get("in_trade"):
            continue
        try:
            try:
                raw    = _ex_pub.fetch_ohlcv(symbol, "1h", limit=1)
                f_high = float(raw[-1][2])
                f_low  = float(raw[-1][3])
            except Exception:
                tk     = _ex_pub.fetch_ticker(symbol)
                f_high = float(tk["high"] or tk["last"])
                f_low  = float(tk["low"]  or tk["last"])

            d  = cs["direction"]
            sp = cs["sl_price"]
            tp = cs["tp_price"]
            ep = cs["entry_price"]
            nt = cs["notional"]
            cap = cs["capital"]

            hit_tp = (f_high >= tp if d == "long" else f_low  <= tp)
            hit_sl = (f_low  <= sp if d == "long" else f_high >= sp)
            if not hit_tp and not hit_sl:
                continue

            xp     = tp if (hit_tp and not hit_sl) else sp
            reason = "TP" if (hit_tp and not hit_sl) else "SL"
            pct    = (xp - ep) / ep if d == "long" else (ep - xp) / ep
            pnl    = max(nt * pct - nt * bb.FEE_RATE * 2, -cap)
            cap   += pnl
            ts_now = datetime.now(timezone.utc)

            rec = dict(timestamp=str(ts_now), coin=coin, direction=d,
                       entry_price=ep, exit_price=round(xp, 6),
                       notional=round(nt, 4), exit_reason=reason,
                       pnl_usdt=round(pnl, 4), capital=round(cap, 4))
            cs["trades"].append(rec)
            _log_trade(rec)
            print(f"  ⚡ INTRABAR {reason:2s}  {coin.upper():5s}  {d.upper()}"
                  f"  xp={xp:.4f}  pnl=${pnl:+.2f}  cap=${cap:.0f}")

            cs["capital"]  = cap
            cs["peak_cap"] = max(cs["peak_cap"], cap)
            cs.update({"in_trade": False, "open_time": None,
                       "em_partial_done": False, "em_trail_active": False})
            params = best.get(coin, {}).get("params", {})
            cs["cooldown_remaining"] = params.get("COOLDOWN_BARS", 0)
            changed = True
        except Exception as e:
            print(f"  [{coin.upper()}] intrabar check error: {e}")
    if changed:
        save_state(state)


# ---- Scheduling --------------------------------------------------------------
def seconds_to_next_candle() -> float:
    now       = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max((next_hour - now).total_seconds() + SLEEP_BUFFER_SEC, 0)


# ---- Entry point -------------------------------------------------------------
def main():
    reset = "--reset" in sys.argv

    print("╔═══════════════════════════════════════════════════════════╗")
    print("║   PAPER TRADING ENGINE — Breakout Strategy v2             ║")
    print(f"║   Testnet : {str(USE_TESTNET):<49}║")
    print(f"║   Capital : ${INITIAL_CAPITAL:,.0f} / coin{' '*(44 - len(f'{INITIAL_CAPITAL:,.0f}'))}║")
    print(f"║   State   : {str(STATE_FILE):<49}║")
    print(f"║   Trades  : {str(TRADE_LOG_FILE):<49}║")
    print("╚═══════════════════════════════════════════════════════════╝\n")

    if not BEST_PARAMS_FILE.exists():
        print(f"ERROR: {BEST_PARAMS_FILE} not found. Run auto_tune first.")
        sys.exit(1)

    if reset:
        STATE_FILE.unlink(missing_ok=True)
        print("State reset.\n")

    best  = json.loads(BEST_PARAMS_FILE.read_text())
    state = load_state()
    ex    = make_exchange()

    while True:
        wait = seconds_to_next_candle()
        nxt  = (datetime.now(timezone.utc) + timedelta(seconds=wait)).strftime("%H:%M:%S UTC")
        print(f"  ⏱  Next cycle in {wait/60:.1f} min  ({nxt})")
        slept = 0
        while slept < wait - 1:
            chunk  = min(INTRABAR_CHECK_INTERVAL, wait - slept)
            time.sleep(chunk)
            slept += chunk
            if slept < wait - 1:
                check_intrabar_sl(ex, state, best)
        best = json.loads(BEST_PARAMS_FILE.read_text())
        run_cycle(ex, state, best)


if __name__ == "__main__":
    main()
