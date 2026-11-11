# -*- coding: utf-8 -*-
"""
历史深度细查 —— 回答"到底能回溯到哪一年, 分钟线能回溯多久"。

运行:  python my_try/06_history_depth.py
"""
import time

import pandas as pd
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS


def wait_ready(api, df, col="close", timeout=20):
    """序列数据随 wait_update 异步到达, 轮询等它非空。"""
    end = time.time() + timeout
    while time.time() < end:
        api.wait_update(deadline=time.time() + 1)
        if len(df.dropna(subset=[col])) > 1:
            return True
    return False


def rng(df, col="close"):
    """返回 (第一根时间, 最后一根时间, 行数)"""
    real = df.dropna(subset=[col])
    d0 = pd.to_datetime(real["datetime"].iloc[0], unit="ns")
    d1 = pd.to_datetime(real["datetime"].iloc[-1], unit="ns")
    return d0, d1, len(real)


api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
try:
    # ---- A. 已下市合约池有多深(按合约代码解析年月) ----
    print("=" * 60)
    print("A. 已下市合约池覆盖的到期年月")
    print("=" * 60)
    for ex, pid in [("SHFE", "cu"), ("DCE", "m"), ("CZCE", "SR"), ("CFFEX", "IF")]:
        ls = api.query_quotes(ins_class="FUTURE", exchange_id=ex,
                              product_id=pid, expired=True)
        print(f"{ex}.{pid:<3} {len(ls):>4} 个: {min(ls)} .. {max(ls)}")

    # ---- B. 池子里最早的合约还有多少历史 ----
    print()
    print("=" * 60)
    print("B. 最早一批已下市合约的数据(日线 / 1分钟)")
    print("=" * 60)
    for sym in ["SHFE.cu2011", "DCE.m2011"]:
        for dur, name in [(86400, "日线"), (60, "1分钟")]:
            try:
                k = api.get_kline_serial(sym, dur, data_length=10000)
                if not wait_ready(api, k):
                    print(f"{sym:<12}{name:<7} 无数据")
                    continue
                d0, d1, n = rng(k)
                print(f"{sym:<12}{name:<7} {d0:%Y-%m-%d %H:%M} ~ {d1:%Y-%m-%d %H:%M}  ({n} 根)")
            except Exception as e:
                print(f"{sym:<12}{name:<7} 失败: {type(e).__name__}: {e}")

    # ---- C. 主连 1 分钟线能回溯多久 ----
    print()
    print("=" * 60)
    print("C. 主连合约 1 分钟线(单次最多 10000 根)覆盖区间")
    print("=" * 60)
    for sym in ["KQ.m@SHFE.cu", "KQ.m@SHFE.rb", "KQ.m@CFFEX.IF"]:
        try:
            k = api.get_kline_serial(sym, 60, data_length=10000)
            if not wait_ready(api, k):
                print(f"{sym:<18} 无数据")
                continue
            d0, d1, n = rng(k)
            print(f"{sym:<18} {d0:%Y-%m-%d %H:%M} ~ {d1:%Y-%m-%d %H:%M}  ({n} 根)")
        except Exception as e:
            print(f"{sym:<18} 失败: {type(e).__name__}: {e}")

    # ---- D. tick 序列能回溯多久 ----
    print()
    print("=" * 60)
    print("D. tick 序列(单次最多 10000 笔)覆盖区间")
    print("=" * 60)
    for sym in ["SHFE.cu2611", "KQ.m@SHFE.rb"]:
        try:
            t = api.get_tick_serial(sym, data_length=10000)
            if not wait_ready(api, t, col="last_price"):
                print(f"{sym:<14} 无 tick")
                continue
            d0, d1, n = rng(t, col="last_price")
            print(f"{sym:<14} {d0:%Y-%m-%d %H:%M:%S} ~ {d1:%Y-%m-%d %H:%M:%S}  ({n} 笔)")
        except Exception as e:
            print(f"{sym:<14} 失败: {type(e).__name__}: {e}")

    # ---- E. 合约静态信息字段抽样 ----
    print()
    print("=" * 60)
    print("E. query_symbol_info 字段抽样(已下市沪铜)")
    print("=" * 60)
    ls = api.query_quotes(ins_class="FUTURE", exchange_id="SHFE",
                          product_id="cu", expired=True)
    info = api.query_symbol_info(ls)
    print(f"共 {len(info.columns)} 列: {list(info.columns)}")
    cols = [c for c in ["instrument_id", "instrument_name", "ins_class",
                        "delivery_year", "delivery_month", "expire_datetime",
                        "volume_multiple", "price_tick"] if c in info.columns]
    print(info[cols].head(2).to_string(index=False))
    print(f"expire_datetime 原始值样例: {info['expire_datetime'].iloc[0]!r}")
finally:
    api.close()
