# -*- coding: utf-8 -*-
"""
数据盘点 —— 用当前账号问服务器: 到底能拿到哪些数据、覆盖多长历史。

运行:  python my_try/05_data_survey.py

分四段打印:
  ① 在交易(未下市)的期货 / 期权合约数量, 按交易所分
  ② 已下市的历史合约数量(能查到的历史合约池)
  ③ 主连合约日线的最早 / 最新一根, 看历史K线能回溯多久
  ④ tick 历史能否下载(试一个很早的日期)
"""
import time
from datetime import date

import pandas as pd
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

EXCHANGES = ["SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX"]
MAINS = ["KQ.m@SHFE.cu", "KQ.m@SHFE.rb", "KQ.m@DCE.m", "KQ.m@DCE.i",
         "KQ.m@CZCE.TA", "KQ.m@CZCE.SR", "KQ.m@CFFEX.IF", "KQ.m@INE.sc",
         "KQ.m@GFEX.si"]

api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
try:
    # ---------- ① 在交易合约 ----------
    print("=" * 62)
    print("① 当前在交易(未下市)合约数量")
    print("=" * 62)
    print(f"{'交易所':<8}{'期货':>8}{'期权':>8}{'指数':>8}{'组合':>8}")
    tot_f = tot_o = 0
    for ex in EXCHANGES:
        n_f = len(api.query_quotes(ins_class="FUTURE", exchange_id=ex, expired=False))
        n_o = len(api.query_quotes(ins_class="OPTION", exchange_id=ex, expired=False))
        n_i = len(api.query_quotes(ins_class="INDEX", exchange_id=ex, expired=False))
        n_c = len(api.query_quotes(ins_class="COMBINE", exchange_id=ex, expired=False))
        tot_f += n_f
        tot_o += n_o
        print(f"{ex:<8}{n_f:>8}{n_o:>8}{n_i:>8}{n_c:>8}")
    n_cont = len(api.query_quotes(ins_class="CONT", expired=False))
    print(f"{'合计':<8}{tot_f:>8}{tot_o:>8}(主连 KQ.m@: {n_cont} 个)")

    # ---------- ② 已下市合约 ----------
    print()
    print("=" * 62)
    print("② 已下市(expired=True)历史合约数量")
    print("=" * 62)
    for ex in EXCHANGES:
        n_f = len(api.query_quotes(ins_class="FUTURE", exchange_id=ex, expired=True))
        n_o = len(api.query_quotes(ins_class="OPTION", exchange_id=ex, expired=True))
        print(f"{ex:<8}期货 {n_f:>7}   期权 {n_o:>7}")

    # 挑几个品种看历史合约池到底有多深
    for ex, pid, label in [("SHFE", "cu", "沪铜"), ("DCE", "m", "豆粕"),
                           ("CZCE", "SR", "白糖"), ("CFFEX", "IF", "沪深300")]:
        ls = api.query_quotes(ins_class="FUTURE", exchange_id=ex,
                              product_id=pid, expired=True)
        if not ls:
            continue
        info = api.query_symbol_info(ls)
        exp = pd.to_datetime(info["expire_datetime"])   # 是字符串, 不能按 ns 解释
        print(f"{ex}.{pid:<3}({label}) 历史合约 {len(ls):>4} 个, "
              f"到期时间 {exp.min():%Y-%m-%d} ~ {exp.max():%Y-%m-%d}")

    # ---------- ③ K线历史深度 ----------
    print()
    print("=" * 62)
    print("③ 主连日线历史深度(data_length=10000)")
    print("=" * 62)
    def wait_ready(df, timeout=15):
        """序列数据是随 wait_update 异步到达的, 轮询等它非空。"""
        end = time.time() + timeout
        while time.time() < end:
            api.wait_update(deadline=time.time() + 1)
            if len(df.dropna(subset=["close"])) > 1:
                return True
        return False

    for sym in MAINS:
        try:
            k = api.get_kline_serial(sym, 86400, data_length=10000)
            if not wait_ready(k):
                print(f"{sym:<18} 超时未取到数据")
                continue
            real = k.dropna(subset=["close"])
            if len(real) == 0:
                print(f"{sym:<18} 无数据")
                continue
            d0 = pd.to_datetime(real["datetime"].iloc[0], unit="ns")
            d1 = pd.to_datetime(real["datetime"].iloc[-1], unit="ns")
            print(f"{sym:<18} {d0:%Y-%m-%d} ~ {d1:%Y-%m-%d}  共 {len(real)} 根日线")
        except Exception as e:
            print(f"{sym:<18} 取数失败: {type(e).__name__}: {e}")

    # ---------- ④ tick 历史 ----------
    print()
    print("=" * 62)
    print("④ tick 历史可用性(试取 2015 年某一天的铜 tick)")
    print("=" * 62)
    try:
        t = api.get_tick_data_series("SHFE.cu1601", start_dt=date(2015, 12, 1),
                                     end_dt=date(2015, 12, 2))
        print(f"2015-12-01 SHFE.cu1601 tick 行数: {len(t)}")
    except Exception as e:
        print(f"取 tick 失败: {type(e).__name__}: {e}")
finally:
    api.close()
