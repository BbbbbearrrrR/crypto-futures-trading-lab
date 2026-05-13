#!/usr/bin/env bash
set -e

PYTHON=/home/bear/Softwares/anaconda3/envs/crypto/bin/python

mkdir -p logs

"$PYTHON" -u paper/paper_trade_breakout.py > logs/paper_breakout.log 2>&1 &
echo "breakout PID: $!"

"$PYTHON" -u paper/paper_trade_calmar.py > logs/paper_calmar.log 2>&1 &
echo "calmar PID: $!"

"$PYTHON" -u paper/paper_trade_martingale.py > logs/paper_martingale.log 2>&1 &
echo "martingale PID: $!"

"$PYTHON" -u paper/paper_trade_regime.py > logs/paper_regime.log 2>&1 &
echo "regime PID: $!"

echo "All four paper trades started. Check logs/ for output."
