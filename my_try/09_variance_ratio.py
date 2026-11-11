# -*- coding: utf-8 -*-
"""
方差比检验 —— 测量"趋势存在于哪个时间尺度", 用来交叉验证参数热力图。

运行:  python my_try/09_variance_ratio.py      (只读数据, 不做回测)

原理 (Lo & MacKinlay, 1988):
    VR(q) = Var(q 期收益) / (q × Var(1 期收益))

      VR(q) = 1   随机游走, 价格没有可利用的记忆
      VR(q) > 1   正自相关 —— 涨了还会涨, 即"趋势型"
      VR(q) < 1   负自相关 —— 涨了会回调, 即"均值回归型"

    统计量 z 在原假设(随机游走)下服从标准正态, |z| > 1.96 即在 5% 水平显著。

为什么必须做这个:
    08 的热力图显示 "fast 8~12 / slow 25~40" 是一片参数高原。但热力图是从
    **收益** 里找规律, 天然有拟合风险 —— 说到底还是在历史收益上挑。
    方差比检验是从 **价格的时序结构** 里测量规律, 两者相互独立。
    如果两种方法指向同一个尺度区间, 结论就非常硬; 如果对不上,
    说明那个"高原"可能仍是拟合出来的。

注意: 这里用的是同方差假设下的 z 统计量(标准做法), 没有做异方差稳健修正。
"""
import time

import numpy as np
import pandas as pd
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
SYMBOL = "KQ.i@SHFE.rb"         # 与 08 保持一致, 结果才能交叉验证
DURATION = 60 * 60              # 1 小时线
DATA_LEN = 8000
QS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40,
      50, 60, 80, 100, 120, 150, 200]
YEAR_QS = [5, 20, 40]           # 分年度检验时看这几个尺度


def vr_stat(logp, q):
    """返回 (VR(q), z 统计量)。logp 是对数价格序列。"""
    r = np.diff(logp)                      # 1 期对数收益
    T = len(r)
    if T <= q + 1:
        return float("nan"), float("nan")
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (T - 1)             # 1 期方差(无偏)
    rq = logp[q:] - logp[:-q]                          # 重叠的 q 期收益
    m = q * (T - q + 1) * (1 - q / T)                  # 重叠修正
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    return float(vr), float(z)


def main():
    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        k = api.get_kline_serial(SYMBOL, DURATION, data_length=DATA_LEN)
        print(f"正在下载 {SYMBOL} 的 {DATA_LEN} 根 {DURATION // 60} 分钟K线 ...")
        last, stable, end = -1, 0, time.time() + 180
        while time.time() < end:
            api.wait_update(deadline=time.time() + 1)
            cnt = int(k["close"].notna().sum())
            if cnt == last and cnt > 100:
                stable += 1
                if stable >= 4:
                    break
            else:
                stable = 0
            last = cnt
        df = k[["datetime", "close"]].dropna().reset_index(drop=True)
    finally:
        api.close()

    if len(df) < 1000:
        raise SystemExit(f"拿到的数据太少({len(df)} 根), 请检查网络后重跑。")

    df["dt"] = pd.to_datetime(df["datetime"], unit="ns") + pd.Timedelta(hours=8)
    logp = np.log(df["close"].to_numpy(dtype=float))
    T = len(logp) - 1

    print()
    print("=" * 104)
    print(f"方差比检验  标的 {SYMBOL}   周期 {DURATION // 60} 分钟")
    print(f"区间 {df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d}   "
          f"{len(logp)} 根K线   1期收益样本数 T = {T}")
    print("=" * 104)
    print(f"{'q(小时)':<9}{'VR(q)':>9}{'z值':>9}{'显著':>6}   偏离 1 的方向和幅度")
    print("-" * 104)

    trending, reverting = [], []
    for q in QS:
        vr, z = vr_stat(logp, q)
        if vr != vr:
            continue
        sig = abs(z) > 1.96
        n = int(round(abs(vr - 1.0) * 60))
        bar = ("+" if vr >= 1 else "-") * min(n, 48)
        flag = "  *" if sig else ""
        print(f"{q:<9}{vr:>9.3f}{z:>+9.2f}{flag:>6}   {bar}")
        if sig and vr > 1:
            trending.append(q)
        elif sig and vr < 1:
            reverting.append(q)

    print()
    print("=" * 104)
    print("结论")
    print("=" * 104)
    print(f"显著趋势 (VR > 1, |z| > 1.96) 的尺度 q: "
          f"{trending if trending else '无'}")
    print(f"显著均值回归 (VR < 1, |z| > 1.96) 的尺度 q: "
          f"{reverting if reverting else '无'}")
    if trending:
        print(f"  -> 趋势集中区间: q ∈ [{min(trending)}, {max(trending)}] 小时")
    print()
    print("与 08 参数热力图的交叉验证:")
    print("  热力图的高原区域是 fast 8~12 / slow 25~40 (小时)。")
    print("  若上面的显著趋势尺度也落在这个量级, 说明两种独立方法互相印证;")
    print("  若对不上, 说明热力图那片'高原'可能仍然是拟合出来的。")

    # ---------- 分年度: 看趋势结构稳不稳定 ----------
    print()
    print("=" * 104)
    print("分年度 VR(q)  —— 带 * 表示该年显著")
    print("=" * 104)
    print(f"{'年份':<8}" + "".join(f"{'q=' + str(q):>14}" for q in YEAR_QS))
    print("-" * 104)
    for year, g in df.groupby(df["dt"].dt.year):
        if len(g) < 4 * max(YEAR_QS):
            print(f"{year:<8}(数据太短, 跳过)")
            continue
        cells = ""
        for q in YEAR_QS:
            vr, z = vr_stat(np.log(g["close"].to_numpy(dtype=float)), q)
            if vr != vr:
                cells += f"{'--':>14}"
            else:
                cells += f"{vr:>9.2f}{'*' if abs(z) > 1.96 else ' '}{'':>4}"
        print(f"{year:<8}{cells}")


if __name__ == "__main__":
    main()
