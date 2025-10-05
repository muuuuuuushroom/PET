#!/usr/bin/env bash
set -euo pipefail

PID=86163         # 要等待的进程号
INTERVAL=5         # 轮询间隔（秒）

# 如果进程存在就等待；不存在就直接执行后续命令
if kill -0 "$PID" 2>/dev/null; then
  echo "PID $PID 正在运行，等待其结束…"
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$INTERVAL"
  done
  echo "PID $PID 已结束。"
else
  echo "PID $PID 当前未在运行，将直接开始后续任务…"
fi

nohup sh train.sh > adapter_s_prob.log 2>&1 &
