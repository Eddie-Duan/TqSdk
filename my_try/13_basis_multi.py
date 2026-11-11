# -*- coding: utf-8 -*-
"""
期限结构研究 · 第三步: 换腿距 + 跨品种

运行:  python my_try/13_basis_multi.py       (只读数据, 不做回测)

前情:
    12_basis_long.py 用 14 段(4.77 年)"主力 - 次主力"价差, 得到:
      * 有界性极稳健: VR(40) = 0.53, 合并 z = -5.90, 14 段全在 1 的同一侧
      * 可预测性极弱且不稳: corr(分位, 未来40h) = -0.089 (r^2 只有 0.8%), 6/14 段反号
      * 成本闸门: 平均 40h 幅度 13.92 元/吨, 一次往返成本 6.0 元/吨
                  需要 71.5% 的方向准确率, 实测约 54.6% -> 每笔亏

本脚本回答两个问题:

问题 1 (换腿距): 会不会"腿距拉宽"能解决成本问题?
    现在: b = P(主力) - P(次主力)        合约差约 4 个月
    改成: b = P(主力) - P(下下次主力)     合约差约 8 个月
    价差的绝对幅度会放大, 而一次往返成本(两腿各开平)不变。
    如果幅度放大得比滑点恶化更快, "幅度/成本"就改善。
    代价: 远月腿流动性差很多 -> 看"流动性比"那一列有没有崩。

问题 2 (跨品种): "价差比价格更有界"是螺纹独有, 还是普遍现象?
    同样方法跑 6 个品种。这是外部效度检验 —— 比在同一个品种上调参数强得多。

【被否掉的方案, 记录在此以免重犯】
    曾经想改成"年化 carry": b = (P主 - P次) / 合约月差。
    这是【空转】: 一个段内合约月差是常数, 整个变换只是"价差 乘 一个正常数"。
    而 VR 尺度无关, 滚动分位只问排序, corr(分位, 未来变化/sd) 里分子分母同比例缩放 ——
    三个统计量全部对这个变换免疫, 结果一个数字都不会变。
    教训: 改函数形式之前, 先问"这个变换真的改变了信息吗"。
"""
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
from scipy.stats import pearsonr
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
# (换月表键, 中文名, 合约乘数 吨/手, 最小变动价位 元/吨)
PRODUCTS = [
    ("SHFE.rb", "螺纹钢", 10, 1.0),
    ("SHFE.hc", "热卷", 10, 1.0),
    ("DCE.i", "铁矿石", 100, 0.5),
    ("DCE.m", "豆粕", 10, 1.0),
    ("CZCE.TA", "PTA", 5, 2.0),
    ("SHFE.cu", "沪铜", 5, 10.0),      # 逐月换月, 预计会被 MIN_SEG_BARS 滤掉
]

LEG_DISTS = [1, 2]          # 1 = 主力-次主力; 2 = 主力-下下次主力
DIST_NAME = {1: "腿距1 (主力-次主力)", 2: "腿距2 (主力-下下次)"}

DURATION = 60 * 60
DATA_LEN = 8000
CONT_TABLE_URL = "https://files.shinnytech.com/continuous_table.json"
START_DATE = date(2022, 1, 1)

VR_QS = [20, 40]            # 只报两个尺度, 表太宽读不动
FWD_H = 40
PCT_WIN = 400
MIN_SEG_BARS = 300
LIQ_WARN = 0.10

# 交易成本: 每条腿 开+平 = 2 次成交; 主力腿滑 1 tick, 次/远月腿滑 2 tick
FILLS_PER_LEG = 2
SLIP_MAIN = 1
SLIP_THIN = 2

POS_LO = 0.10               # 分位 <= 此值 -> 做多价差
POS_HI = 0.90               # 分位 >= 此值 -> 做空价差
PL_HORIZONS = [40, 160]     # 朴素信号的两个持有期

REPORT_QS = VR_QS


def pad_cn(text, width):
    """按终端显示宽度补齐(中文按 2 计)。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - w)


# ==================== 通用统计 ====================
def vr_stat(x, q):
    """Lo-MacKinlay 方差比(尺度无关)。VR < 1 = 均值回归。"""
    r = np.diff(x)
    T = len(r)
    if T <= q + 1:
        return float("nan"), float("nan")
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (T - 1)
    rq = x[q:] - x[:-q]
    m = q * (T - q + 1) * (1 - q / T)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    return float(vr), float(z)


def rolling_pctile(v, w):
    """当前值在过去 w 个样本中的分位(0~1), 只用历史。"""
    out = np.full(len(v), np.nan)
    for i in range(w, len(v)):
        win = v[i - w + 1:i + 1]
        out[i] = float((win <= v[i]).mean())
    return out


def seg_window(s, n, win=400):
    """每段自己的滚动分位窗口。"""
    return min(win, max(30, n // 3))


def fwd_change(spread, h, sd):
    """未来 h 根的变化, 除以该段标准差 -> 可比单位。"""
    out = np.full(len(spread), np.nan)
    if len(spread) > h + 1:
        out[:-h] = (spread[h:] - spread[:-h]) / sd
    return out


# ==================== 数据获取 ====================
def get_rollovers(table, key):
    rows = table.get(key)
    if not rows:
        return None
    out = [(datetime.strptime(d, "%Y%m%d").date(), sym) for d, sym in rows]
    out.sort(key=lambda x: x[0])
    return out


def build_pairs(rollovers, start, dist):
    """构造 [(段起始, 段结束, 近腿, 远腿), ...]; 段窗口永远是[本次换月, 下次换月)。"""
    out = []
    for i in range(len(rollovers) - dist):
        seg_start, near = rollovers[i]
        seg_end = rollovers[i + 1][0]
        far = rollovers[i + dist][1]
        if seg_end <= start:
            continue
        out.append((seg_start, seg_end, near, far))
    return out


def wait_all(api, frames, timeout=600):
    """等所有序列数据稳定。"""
    last, stable, end = -1, 0, time.time() + timeout
    while time.time() < end:
        api.wait_update(deadline=time.time() + 1)
        cnt = sum(int(f["close"].notna().sum()) for f in frames)
        if cnt == last and cnt > 100:
            stable += 1
            if stable >= 5:
                return
        else:
            stable = 0
        last = cnt


# ==================== 逐段构造 ====================
def make_segments(pairs, data, tick):
    """把 [(段起, 段止, 近腿, 远腿)] 变成可统计的段列表。裁剪到段窗口内。"""
    cost_yt = 2 * (SLIP_MAIN + SLIP_THIN) * tick
    segs = []
    for i, (s, e, near, far) in enumerate(pairs, 1):
        dn, df = data.get(near), data.get(far)
        if dn is None or df is None or len(dn) < 50 or len(df) < 50:
            continue
        j = pd.merge(dn.rename(columns={"close": "a", "volume": "va"}),
                     df.rename(columns={"close": "b", "volume": "vb"}),
                     on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)
        # 只保留 [段起始, 段结束)。不加这一步会混进"远腿已经变成近腿之后"的数据。
        dt = pd.to_datetime(j["datetime"], unit="ns") + pd.Timedelta(hours=8)
        keep = (dt >= pd.Timestamp(s)) & (dt < pd.Timestamp(e))
        j = j[keep].reset_index(drop=True)
        dt = dt[keep].reset_index(drop=True)
        if len(j) < MIN_SEG_BARS:
            continue
        spread = (j["a"] - j["b"]).to_numpy(dtype=float)
        sd = float(spread.std())
        if not (sd > 0):
            continue
        liq = float(j["vb"].mean() / j["va"].mean()) if j["va"].mean() > 0 else 0.0
        segs.append({"i": i, "near": near, "far": far, "spread": spread, "sd": sd,
                     "dt": dt, "n": len(spread), "liq": liq})
    return segs, cost_yt


# ==================== 三项检验 ====================
def pooled_vr(segs, q):
    vals, ws, ses = [], [], []
    for s in segs:
        T = s["n"]
        if T <= q + 2:
            continue
        vr, _ = vr_stat(s["spread"], q)
        if vr != vr:
            continue
        vals.append(vr)
        ws.append(T)
        ses.append(np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T)))
    if not vals:
        return float("nan"), float("nan")
    w = np.array(ws, dtype=float)
    pooled = float(np.average(vals, weights=w))
    se = float(np.sqrt(np.sum(w ** 2 * np.array(ses) ** 2)) / w.sum())
    return pooled, (pooled - 1) / se


def pooled_corr(segs):
    xs, ys = [], []
    for s in segs:
        pct = rolling_pctile(s["spread"], seg_window(s, s["n"]))
        fwd = fwd_change(s["spread"], FWD_H, s["sd"])
        ok = ~np.isnan(pct) & ~np.isnan(fwd)
        xs.append(pct[ok])
        ys.append(fwd[ok])
    if not xs:
        return float("nan"), float("nan")
    X, Y = np.concatenate(xs), np.concatenate(ys)
    if len(X) < 50:
        return float("nan"), float("nan")
    r, p = pearsonr(X, Y)
    return float(r), float(p)


def n_significant(segs):
    """多少个段单独算出显著负相关(成立)。"""
    good = 0
    for s in segs:
        pct = rolling_pctile(s["spread"], seg_window(s, s["n"]))
        fwd = fwd_change(s["spread"], FWD_H, s["sd"])
        ok = ~np.isnan(pct) & ~np.isnan(fwd)
        if ok.sum() < 50:
            continue
        rr, pp = pearsonr(pct[ok], fwd[ok])
        if rr < 0 and pp < 0.05:
            good += 1
    return good


def naive_pnl(segs, h, cost_yt):
    """规则写死、不调参、非重叠。返回 (每笔毛利, 一次往返成本, 笔数, 胜率)。"""
    gross, n, wins = [], 0, 0
    for s in segs:
        if s["n"] <= h + 60:
            continue
        pct = rolling_pctile(s["spread"], seg_window(s, s["n"]))
        i = seg_window(s, s["n"])
        while i < len(pct) - h:
            if np.isnan(pct[i]):
                i += 1
                continue
            pos = 1 if pct[i] <= POS_LO else (-1 if pct[i] >= POS_HI else 0)
            if pos == 0:
                i += 1
                continue
            mv = (s["spread"][i + h] - s["spread"][i]) * pos
            gross.append(mv)
            n += 1
            wins += int(mv > 0)
            i += h
    if n == 0:
        return float("nan"), cost_yt, 0, float("nan")
    return float(np.mean(gross)), cost_yt, n, wins / n * 100


def summarize(pairs, data, tick):
    segs, cost_yt = make_segments(pairs, data, tick)
    if not segs:
        return None
    nbars = sum(s["n"] for s in segs)
    first = min(s["dt"].iloc[0] for s in segs)
    last = max(s["dt"].iloc[-1] for s in segs)
    rec = {"nseg": len(segs), "nbars": nbars,
           "span": (last - first).days / 365.25,
           "liq_min": min(s["liq"] for s in segs),
           "sd_cost": float(np.median([s["sd"] / cost_yt for s in segs])),
           "cost": cost_yt,
           "vr": {}, "corr": pooled_corr(segs), "nsig": n_significant(segs),
           "pl": {}}
    for q in REPORT_QS:
        rec["vr"][q] = pooled_vr(segs, q)
    for h in PL_HORIZONS:
        rec["pl"][h] = naive_pnl(segs, h, cost_yt)
    return rec


# ==================== 主流程 ====================
def main():
    print("=" * 112)
    print("期限结构 · 换腿距 + 跨品种         (只读数据, 不做回测)")
    print("=" * 112)

    table = requests.get(CONT_TABLE_URL, timeout=30).json()
    print(f"换月表已下载, 共 {len(table)} 个品种键")

    results = {}      # (key, dist) -> rec
    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        for key, name, mult, tick in PRODUCTS:
            t0 = time.time()
            rollovers = get_rollovers(table, key)
            if rollovers is None:
                print(f"\n[{name}] {key}: 换月表里没有这个键, 跳过")
                continue

            # 两个腿距用到的合约并集(其实就是所有出现过的主力)
            need = set()
            for dist in LEG_DISTS:
                for _, _, near, far in build_pairs(rollovers, START_DATE, dist):
                    need.add(near)
                    need.add(far)
            need = sorted(need)
            print(f"\n[{name}] {key}: 下载 {len(need)} 个合约的K线 ...")

            frames = {s: api.get_kline_serial(s, DURATION, data_length=DATA_LEN) for s in need}
            wait_all(api, list(frames.values()))
            data = {s: frames[s][["datetime", "close", "volume"]].dropna().reset_index(drop=True)
                    for s in need}
            del frames
            print(f"  完成, 用时 {time.time() - t0:.0f}s")

            for dist in LEG_DISTS:
                pairs = build_pairs(rollovers, START_DATE, dist)
                results[(key, dist)] = summarize(pairs, data, tick)
    finally:
        api.close()

    # ---------------- 表 1: 统计性质 ----------------
    print()
    print("=" * 112)
    print("表 1  统计性质")
    print("  VR = 方差比(尺度无关, <1 表示价差有界/均值回归);  z = 跨段合并的 z 值")
    print("  corr = 滚动分位 vs 未来40h价差变化;  成立段 = 单独显著负相关的段数")
    print("=" * 112)
    hdr = (pad_cn("品种", 10) + pad_cn("腿距", 6) + f"{'段数':>5}{'根数':>7}{'跨度':>7}"
           f"{'VR20':>8}{'z20':>8}{'VR40':>8}{'z40':>8}"
           f"{'corr':>9}{'p值':>11}{'成立段':>9}{'最差流动性':>11}")
    print(hdr)
    print("-" * 112)
    for key, name, mult, tick in PRODUCTS:
        for dist in LEG_DISTS:
            rec = results.get((key, dist))
            if rec is None:
                print(f"{pad_cn(name, 10)}{pad_cn('腿距' + str(dist), 6)}  段长为空(被滤掉), 无结果")
                continue
            v20, z20 = rec["vr"][20]
            v40, z40 = rec["vr"][40]
            r, p = rec["corr"]
            print(f"{pad_cn(name, 10)}{pad_cn('腿距' + str(dist), 6)}"
                  f"{rec['nseg']:>5}{rec['nbars']:>7}{rec['span']:>7.2f}"
                  f"{v20:>8.3f}{z20:>8.2f}{v40:>8.3f}{z40:>8.2f}"
                  f"{r:>+9.4f}{p:>11.3g}{rec['nsig']:>6}/{rec['nseg']:<3}"
                  f"{rec['liq_min']:>11.2f}")

    # ---------------- 表 2: 朴素信号能不能赚钱 ----------------
    print()
    print("=" * 112)
    print("表 2  朴素信号 (分位<=10% 做多价差 / >=90% 做空; 持有 H 根后无条件平; 非重叠)")
    print("  每笔净利 = 每笔毛利 - 一次往返成本(两条腿各开平, 近腿滑1tick/远腿滑2tick)")
    print("  净利/成本 > 0 才叫赚钱。这个比值跨品种可比。")
    print("=" * 112)
    hdr2 = (pad_cn("品种", 10) + pad_cn("腿距", 6) + f"{'中位 空间/成本':>15}"
            + f"{'H=40 毛利/成本':>16}{'笔数':>7}{'胜率':>8}"
            + f"{'H=160 毛利/成本':>17}{'笔数':>7}{'胜率':>8}")
    print(hdr2)
    print("-" * 112)
    for key, name, mult, tick in PRODUCTS:
        for dist in LEG_DISTS:
            rec = results.get((key, dist))
            if rec is None:
                continue
            g40, c40, n40, w40 = rec["pl"][40]
            g160, c160, n160, w160 = rec["pl"][160]
            def ratio(g, c):
                return g / c if c > 0 and g == g else float("nan")
            print(f"{pad_cn(name, 10)}{pad_cn('腿距' + str(dist), 6)}"
                  f"{rec['sd_cost']:>14.1f}x"
                  f"{ratio(g40, c40):>15.2f}{n40:>7}{w40:>7.1f}%"
                  f"{ratio(g160, c160):>16.2f}{n160:>7}{w160:>7.1f}%")
    print()
    print("  毛利/成本 < 1 -> 亏。比值越大越好, > 1 才谈得上正期望。")
    print("  中位 空间/成本 = 该品种各段价差标准差的中位数 / 一次往返成本;")
    print("    低于 ~5 说明价差自己都懒得动, 别谈交易。")


if __name__ == "__main__":
    main()
