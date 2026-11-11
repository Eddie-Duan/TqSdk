# -*- coding: utf-8 -*-
"""
期限结构长样本检验 —— 把样本从 0.67 年扩到 4 年。

运行:  python my_try/12_basis_long.py      (只读数据, 不做回测)

为什么需要这个:
    11_basis_test.py 在 rb2610-rb2701 这一对上得到了很强的均值回归结论
    (VR(40)=0.249, corr=-0.241, 都显著)。但那一对只有 0.67 年, 而且期间
    Contango 占比 100% —— 只在一种行情状态下检验过, 统计效力弱。
    我们已经被"5 个月样本"坑过一次(04 -> 08), 不能在同一个地方再摔。

怎么扩样本:
    单个合约只有约 11 个月生命周期, 但可以拼接"主力 - 次主力"合约对:
        从换月表拿到所有主力变更点, 每个区间内:
            主力  = 该区间的主力合约
            次主力 = 下一个区间的主力合约(它正好是当时的次主力)
        每一段都有一个真实的期限结构价差 P(主力) - P(次主力)
    把这些段接起来, 就得到 4 年的期限结构序列。

关键细节(方法自洽性):
    不同月份组合的价差绝对水平不可比(rb2605-rb2610 和 rb2610-rb2701 量纲不同),
    所以**绝不能拼原始价差**。本脚本的做法:
      * 每一段内部独立计算滚动分位(尺度无关)和标准差
      * 未来变化统一除以该段价差的标准差 -> 变成"几个标准差"的可比单位
      * 方差比 VR 本身是尺度无关的, 可以分段算再加权平均
    这样拼接不会引入人为跳变。

两个必须做对的细节(第一版没做对, 见下):
  1) 【裁剪】get_kline_serial 返回的是每个合约的**整个生命周期**。inner merge 后
     重叠区间会是"次主力上市 ~ 主力退市", 把"次主力已经变成主力之后"的数据也带进来 ——
     那段里 (m, sec) 已经不是"主力-次主力", 是过期的合约对。必须裁到 [段起始, 段结束)。
     症状: 报出的总年数约为实际跨度的 2 倍(14 段跨 4.8 年却报 9.2 年)。
  2) 【成本】VR<1 只说明价差"有界", 不说明"可预测"。买卖价差跳动(bid-ask bounce)
     本身就会造成 VR<1。所以必须把波动换算成 元/吨, 和一次往返成本比一比。

注意: 成交仍是简化假设; 本脚本不下任何单, 只做统计检验。
"""
import json
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
PRODUCT_KEY = "SHFE.rb"                 # 换月表里的品种键
SYM_PREFIX = "SHFE.rb"                  # 合约代码前缀
DURATION = 60 * 60
DATA_LEN = 8000                         # 够覆盖单个合约的完整生命周期
VOLUME_MULTIPLE = 10                    # 螺纹 10 吨/手
CONT_TABLE_URL = "https://files.shinnytech.com/continuous_table.json"
START_DATE = date(2022, 1, 1)           # 从哪一年开始拼接
VR_QS = [5, 20, 40, 80, 160]
FWD_H = 40                              # 未来 H 小时
PCT_WIN = 400                           # 滚动分位窗口(每段内部单独算)
MIN_SEG_BARS = 300                      # 段太短就不参与统计(裁剪后一段约 300~900 根)
LIQ_WARN = 0.10                         # 次主力成交量/主力 低于此值就告警

# 交易成本(用来判断"这点波动够不够付过路费")
TICK = 1.0                              # 螺纹最小变动价位 1 元/吨
FILLS_PER_LEG = 2                       # 每条腿 开 + 平
SLIP_TICKS = 1                          # 主力腿每次成交滑 1 个 tick
SLIP_TICKS_THIN = 2                     # 次主力腿成交量只有主力的 ~10%, 保守给 2 个 tick
COST_YT = TICK * FILLS_PER_LEG * (SLIP_TICKS + SLIP_TICKS)                # = 4.0 元/吨, 乐观口径
COST_YT_THIN = TICK * FILLS_PER_LEG * (SLIP_TICKS + SLIP_TICKS_THIN)      # = 6.0 元/吨, 现实口径

# 朴素信号(规则写死, 不调参; 只是为了看"最朴素的用法能不能赚钱")
HORIZONS = [10, 20, 40, 80, 160, 320]   # 持仓期扫描(小时)
POS_LO = 0.10                           # 分位 <= 此值 -> 做多价差(买近抛远)
POS_HI = 0.90                           # 分位 >= 此值 -> 做空价差


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


def get_rollovers():
    """下载全量换月表, 返回 [(date, 主力合约), ...] 按时间升序。"""
    rsp = requests.get(CONT_TABLE_URL, timeout=30)
    rsp.raise_for_status()
    rows = rsp.json()[PRODUCT_KEY]
    out = []
    for d, sym in rows:
        out.append((datetime.strptime(d, "%Y%m%d").date(), sym))
    out.sort(key=lambda x: x[0])
    return out


def build_pairs(rollovers, start):
    """构造 [(段起始, 段结束, 主力, 次主力), ...]"""
    pairs = []
    for i in range(len(rollovers) - 1):
        seg_start, main = rollovers[i]
        seg_end, second = rollovers[i + 1]
        if seg_end <= start:
            continue
        pairs.append((seg_start, seg_end, main, second))
    return pairs


def wait_all(api, frames, timeout=300):
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


def main():
    print("=" * 108)
    print("期限结构长样本检验   标的: 螺纹钢 SHFE.rb")
    print(f"拼接方式: 主力 - 次主力(下一个主力)   起点 {START_DATE}")
    print("=" * 108)

    rollovers = get_rollovers()
    pairs = build_pairs(rollovers, START_DATE)
    if not pairs:
        raise SystemExit("换月表里没有可用的区间。")

    print(f"\n构造出 {len(pairs)} 个合约对:")
    print(f"  {'#':<4}{'起':<12}{'止':<12}{'主力':<14}{'次主力':<14}")
    print("  " + "-" * 60)
    for i, (s, e, m, sec) in enumerate(pairs, 1):
        print(f"  {i:<4}{s!s:<12}{e!s:<12}{m:<14}{sec:<14}")

    needed = sorted({m for _, _, m, _ in pairs} | {sec for _, _, _, sec in pairs})
    print(f"\n需要下载 {len(needed)} 个合约的K线 ...")

    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        frames = {s: api.get_kline_serial(s, DURATION, data_length=DATA_LEN) for s in needed}
        wait_all(api, list(frames.values()))
        data = {}
        for s in needed:
            d = frames[s][["datetime", "close", "volume"]].dropna().reset_index(drop=True)
            data[s] = d
    finally:
        api.close()

    # ---------- 逐段构造价差 ----------
    segs = []
    print("\n" + "=" * 108)
    print("逐段结果")
    print("=" * 108)
    print(f"  {'#':<4}{'主力':<14}{'次主力':<14}{'根数':>7}{'价差均值':>10}"
          f"{'标准差':>9}{'Contango':>10}{'流动性比':>10}{'空间/成本':>11}")
    print("  " + "-" * 106)

    for i, (s, e, m, sec) in enumerate(pairs, 1):
        dm, ds = data.get(m), data.get(sec)
        if dm is None or ds is None or len(dm) < 50 or len(ds) < 50:
            print(f"  {i:<4}{m:<14}{sec:<14}  无数据, 跳过")
            continue
        j = pd.merge(dm.rename(columns={"close": "a", "volume": "va"}),
                     ds.rename(columns={"close": "b", "volume": "vb"}),
                     on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)
        # 【裁剪】只保留 [段起始, 段结束)。不加这一步, 收进来的就是
        # "次主力上市 ~ 主力退市" 这段重叠期, 尾部已经不是主力-次主力了。
        dt = pd.to_datetime(j["datetime"], unit="ns") + pd.Timedelta(hours=8)
        keep = (dt >= pd.Timestamp(s)) & (dt < pd.Timestamp(e))
        j = j[keep].reset_index(drop=True)
        dt = dt[keep].reset_index(drop=True)
        if len(j) < MIN_SEG_BARS:
            print(f"  {i:<4}{m:<14}{sec:<14}{len(j):>7}  裁剪后太短, 跳过")
            continue
        spread = (j["a"] - j["b"]).to_numpy(dtype=float)
        sd = float(spread.std())
        if not (sd > 0):
            continue
        liq = float(j["vb"].mean() / j["va"].mean()) if j["va"].mean() > 0 else float("nan")
        flag = "  <== 流动性偏低" if liq < LIQ_WARN else ""
        segs.append({"i": i, "main": m, "second": sec, "spread": spread, "sd": sd,
                     "dt": dt, "n": len(spread), "liq": liq})
        print(f"  {i:<4}{m:<14}{sec:<14}{len(j):>7}{spread.mean():>10.1f}"
              f"{sd:>9.1f}{(spread < 0).mean() * 100:>9.1f}%{liq:>10.2f}"
              f"{sd / COST_YT:>10.1f}x{flag}")

    if not segs:
        raise SystemExit("没有可用区间。")

    total_n = sum(s["n"] for s in segs)
    first_dt = min(s["dt"].iloc[0] for s in segs)
    last_dt = max(s["dt"].iloc[-1] for s in segs)
    span_y = (last_dt - first_dt).days / 365.25
    sum_y = sum((s["dt"].iloc[-1] - s["dt"].iloc[0]).days for s in segs) / 365.25
    print(f"\n  合计 {len(segs)} 段, {total_n} 根K线, 覆盖 {first_dt:%Y-%m-%d} ~ "
          f"{last_dt:%Y-%m-%d}, 实际跨度约 {span_y:.2f} 年")
    print(f"  各段时长之和 {sum_y:.2f} 年 (应接近实际跨度; 明显偏大说明有段重叠/污染)")
    print(f"  流动性比 = 次主力日均成交量 / 主力日均成交量;"
          f" 低于 {LIQ_WARN} 说明次主力太冷门, 实际难以成交")
    print(f"  空间/成本 = 该段价差标准差 / 一次往返成本({COST_YT:.1f} 元/吨);"
          f" 低于 ~5 基本不用谈交易")

    # ---------- 合并: 方差比(尺度无关, 分段算再加权) ----------
    print()
    print("=" * 108)
    print("合并检验 1: 价差的方差比 (各段加权平均)")
    print("=" * 108)
    print(f"  {'q(小时)':<9}{'加权VR':>9}{'合并z':>9}{'标准误':>9}{'显著段数':>10}")
    print("  " + "-" * 96)
    for q in VR_QS:
        vals, ws, ses, sig = [], [], [], 0
        for s in segs:
            T = s["n"]
            if T <= q + 2:
                continue
            vr, z = vr_stat(s["spread"], q)
            if vr != vr:
                continue
            vals.append(vr)
            ws.append(T)
            # 单个区间的 VR 标准误(同方差形式)。T 小的时候这个值很大,
            # 所以单段几乎测不出东西 —— 不能靠"显著段数"下结论。
            ses.append(np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T)))
            if abs(z) > 1.96 and vr < 1:
                sig += 1
        if not vals:
            continue
        w = np.array(ws, dtype=float)
        pooled = float(np.average(vals, weights=w))
        se = float(np.sqrt(np.sum(w ** 2 * np.array(ses) ** 2)) / w.sum())
        zpool = (pooled - 1) / se
        print(f"  {q:<9}{pooled:>9.3f}{zpool:>9.2f}{se:>9.3f}{sig:>7}/{len(vals):<3}"
              f"   {'显著' if abs(zpool) > 1.96 else '不显著'}")
        print("   各段 VR: " + ", ".join(f"{v:.2f}" for v in vals))

    print("\n  合并z = 把各独立区间的 VR 估计按精度加权合并后的 z 统计量(这才是该看的数)")
    print("  各段 VR 那一行的作用: 看 14 个数字是不是都在 1 的同一侧、量级是否一致")

    # ---------- 合并: 滚动分位 -> 未来变化(标准化) ----------
    rows = []
    for s in segs:
        w = min(PCT_WIN, max(30, s["n"] // 3))
        pct = rolling_pctile(s["spread"], w)
        fwd = np.full(s["n"], np.nan)
        if s["n"] > FWD_H + 1:
            fwd[:-FWD_H] = (s["spread"][FWD_H:] - s["spread"][:-FWD_H]) / s["sd"]
        ok = ~np.isnan(pct) & ~np.isnan(fwd)
        f = fwd[ok]
        # 去掉该段的系统性漂移(螺纹 4 年从 Backwardation 转 Contango, 价差整体下移)。
        # 不改这一步的话, 分位表每一行都会被这个漂移压成负数, 读不出形状。
        # 注意: Pearson 相关对 y 平移不变, 所以下面的 corr 不受影响, 无需去均值。
        f = f - f.mean()
        rows.append(pd.DataFrame({"seg": f"#{s['i']}", "pct": pct[ok], "fwd": f}))
    allrows = pd.concat(rows, ignore_index=True)

    print()
    print("=" * 108)
    print(f"合并检验 2: 滚动分位 -> 未来 {FWD_H} 小时价差变化")
    print(f"  (单位: 该段价差标准差的倍数; 每段已去均值, 去掉 4 年 Backwardation->Contango 的系统漂移)")
    print("=" * 108)
    print(f"  {'当前分位':<14}{'样本数':>9}{'未来变化均值':>15}{'反向占比':>11}")
    print("  " + "-" * 96)
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.001]
    labels = ["0-10% 极低", "10-25%", "25-50%", "50-75%", "75-90%", "90-100% 极高"]
    allrows["g"] = pd.cut(allrows["pct"], bins, labels=labels, include_lowest=True)
    for gi, part in allrows.groupby("g", observed=True):
        rev = (part["fwd"] > 0).mean() if str(gi).startswith("0-10") or str(gi).startswith("10-25") \
            else (part["fwd"] < 0).mean()
        print(f"  {str(gi):<14}{len(part):>9}{part['fwd'].mean():>+14.3f}"
              f"{rev * 100:>10.1f}%")

    from scipy.stats import pearsonr
    r, pval = pearsonr(allrows["pct"], allrows["fwd"])
    print()
    print(f"  corr(当前分位, 未来{FWD_H}h价差变化) = {r:+.4f}   p = {pval:.4g}   "
          f"({'显著' if pval < 0.05 else '不显著'})")
    print(f"  对照 11 的结果: A 对 = -0.2408 (0.67年), B 代理 = -0.2870 (4.74年)")

    # ---------- 成本闸门: 这点波动够不够付过路费 ----------
    # 注意写法: 要的是"隔 FWD_H 步的差", 所以用切片;
    # np.diff(x, n=FWD_H) 是 40 阶差分(连续差分 40 次), 系数爆炸, 得到天文数字。
    def abs_move(s):
        return float(np.abs(s["spread"][FWD_H:] - s["spread"][:-FWD_H]).mean())

    am = float(np.mean([abs_move(s) for s in segs if s["n"] > FWD_H]))
    print()
    print("=" * 108)
    print("成本闸门 —— 价差自己愿意动的幅度, 够不够付一次往返的过路费")
    print("=" * 108)
    print(f"  各段 |未来 {FWD_H}h 价差变化| 的平均值   = {am:8.2f} 元/吨")
    print()
    print(f"  {'成本口径':<36}{'一次往返':>10}{'净赚上限':>10}{'盈亏平衡胜率':>14}")
    print("  " + "-" * 70)
    for name, c in (("乐观: 两腿各滑 1 tick", COST_YT),
                    ("现实: 主力腿 1 tick + 次主力腿 2 tick", COST_YT_THIN)):
        be = 0.5 + c / (2 * am) if am > 0 else float("nan")
        print(f"  {name:<36}{c:>9.2f}{am - c:>10.2f}{be * 100:>13.1f}%")
    print(f"\n  (公式: p*{am:.1f} - (1-p)*{am:.1f} - 成本 > 0   ->   p > 0.5 + 成本/(2*幅度))")
    print(f"  拿这个胜率去对照上面的逐段一致性: 实际的方向判对率是多少?")

    # ---------- 逐段一致性 ----------
    print()
    print("=" * 108)
    print("逐段一致性 —— 每个段单独算 corr, 看是否普遍成立(而不是被某一段拉出来的)")
    print("=" * 108)
    print(f"  {'#':<4}{'主力':<14}{'根数':>7}{'corr':>10}{'p值':>10}"
          f"{'结论':>10}{'40h幅度':>10}{'幅度/成本':>11}")
    print("  " + "-" * 88)
    pos = 0
    for s in segs:
        w = min(PCT_WIN, max(30, s["n"] // 3))
        pct = rolling_pctile(s["spread"], w)
        fwd = np.full(s["n"], np.nan)
        if s["n"] > FWD_H + 1:
            fwd[:-FWD_H] = (s["spread"][FWD_H:] - s["spread"][:-FWD_H]) / s["sd"]
        ok = ~np.isnan(pct) & ~np.isnan(fwd)
        if ok.sum() < 50:
            continue
        rr, pp = pearsonr(pct[ok], fwd[ok])
        good = rr < 0 and pp < 0.05
        pos += int(good)
        amv = abs_move(s)
        print(f"  {s['i']:<4}{s['main']:<14}{s['n']:>7}{rr:>+10.3f}{pp:>10.4f}"
              f"{'  成立' if good else '  不显著' if rr < 0 else '  方向相反':>10}"
              f"{amv:>9.2f}{amv / COST_YT:>10.1f}x")
    print(f"\n  {pos}/{len(segs)} 段呈现显著均值回归。若多数段都成立 -> 是普遍规律; 若只是个别的 -> 要警惕。")
    print(f"  关键对照: 看「成立」和「方向相反」的段, 在「幅度/成本」上有没有区别。")
    print(f"  如果方向对的段恰好都是幅度小到付不起成本的段, 那这个信号就没有商业价值。")

    # ---------- 把上面的估算变成真事: 朴素信号的原始盈亏 ----------
    # 前面是统计量。统计量只是旁证, 直接算钱才是主证。
    # 规则写死、不调参、非重叠开仓: 只为了回答"最朴素的用法能不能赚钱"。
    print()
    print("=" * 108)
    print("朴素信号的原始盈亏 (规则写死, 不调参) —— 直接算钱, 不再看相关系数")
    print(f"  开仓: 分位<={POS_LO:.0%} 做多价差(买近月/抛远月); 分位>={POS_HI:.0%} 做空价差; 其余空仓")
    print(f"  平仓: 持有 H 根K线后无条件平掉, 期间不重复开仓(非重叠)")
    print(f"  单位: 元/吨 (1 手螺纹 = 10 吨, 所以乘 10 就是元/手)")
    print("=" * 108)
    print(f"  {'持有期H':>8}{'交易笔数':>10}{'每笔毛利':>11}{'每笔净利':>11}{'方向胜率':>10}"
          f"{'净利合计':>11}{'结论':>8}")
    print("  " + "-" * 86)
    for h in HORIZONS:
        gross, n_tr, wins = [], 0, 0
        for s in segs:
            if s["n"] <= h + 60:
                continue
            w = min(PCT_WIN, max(30, s["n"] // 3))
            pct = rolling_pctile(s["spread"], w)
            i = w
            while i < len(pct) - h:
                if np.isnan(pct[i]):
                    i += 1
                    continue
                p = 1 if pct[i] <= POS_LO else (-1 if pct[i] >= POS_HI else 0)
                if p == 0:
                    i += 1
                    continue
                mv = (s["spread"][i + h] - s["spread"][i]) * p
                gross.append(mv)
                n_tr += 1
                wins += int(mv > 0)
                i += h                      # 持仓 h 根, 期间不开新仓
        if n_tr == 0:
            continue
        gm = float(np.mean(gross))
        nm = gm - COST_YT_THIN
        print(f"  {h:>8}{n_tr:>10}{gm:>+11.2f}{nm:>+11.2f}{wins / n_tr * 100:>9.1f}%"
              f"{nm * n_tr:>+11.1f}{'  亏' if nm < 0 else '  盈'}")
    print()
    print(f"  每笔净利 = 每笔毛利 - 现实口径成本({COST_YT_THIN:.1f} 元/吨)")
    print(f"  若各持有期每笔净利都是负的, 那这个信号就是不可交易的 —— 与相关系数显著与否无关。")


if __name__ == "__main__":
    main()
