# -*- coding: utf-8 -*-
"""
历史能回溯多久 —— 直接回答"能不能用更早的历史做分析"。

运行:  python my_try/07_hist_range.py      (只读数据, 不做回测)

原理: 非回测模式下 get_kline_serial 会返回"最近 N 根"K线(N 最多 8000~10000)。
      所以每根序列的"最早时间"就是该合约/该周期能回溯到的最远处。

对比三类标的:
  * 主连 KQ.m@SHFE.rb  —— 连续价格, 不过期, 但不能直接交易
  * 指数 KQ.i@SHFE.rb  —— 连续价格, 不过期, 不能交易（更平滑）
  * 真实合约           —— 能交易, 但受"上市~交割"约 12 个月的生命周期限制

注意: 打印的时间是 UTC(= 北京时间 - 8 小时), 看日期即可。
"""
import time

import pandas as pd
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

TARGETS = [
    # (合约,      K线周期秒,  说明)
    ("KQ.m@SHFE.rb", 86400, "主连 日线"),
    ("KQ.m@SHFE.rb", 3600, "主连 1小时"),
    ("KQ.i@SHFE.rb", 86400, "指数 日线"),
    ("KQ.i@SHFE.rb", 3600, "指数 1小时"),
    ("SHFE.rb2610", 86400, "真实合约 日线"),
    ("SHFE.rb2610", 3600, "真实合约 1小时"),
    ("SHFE.rb2510", 3600, "老合约 1小时"),
]


def _range(api, symbol, dur, timeout=30):
    """取序列并等数据到达, 返回 (最早, 最新, 根数)"""
    k = api.get_kline_serial(symbol, dur, data_length=8000)
    end = time.time() + timeout
    while time.time() < end:
        api.wait_update(deadline=time.time() + 1)
        if len(k.dropna(subset=["close"])) > 1:
            break
    real = k.dropna(subset=["close"])
    if len(real) < 2:
        return None, None, 0
    d0 = pd.to_datetime(real["datetime"].iloc[0], unit="ns")
    d1 = pd.to_datetime(real["datetime"].iloc[-1], unit="ns")
    return d0, d1, len(real)


api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
try:
    print("=" * 78)
    print(f"{'合约':<16}{'周期':<12}{'最早':<14}{'最新':<14}{'根数':>7}")
    print("=" * 78)
    for sym, dur, label in TARGETS:
        try:
            d0, d1, n = _range(api, sym, dur)
            if d0 is None:
                print(f"{sym:<16}{label:<12}{'无数据':<14}")
            else:
                print(f"{sym:<16}{label:<12}{d0:%Y-%m-%d}    {d1:%Y-%m-%d}    {n:>7}")
        except Exception as e:
            print(f"{sym:<16}{label:<12}失败: {type(e).__name__}: {e}")
finally:
    api.close()
