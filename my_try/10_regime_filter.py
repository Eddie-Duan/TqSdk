# -*- coding: utf-8 -*-
"""
Regime 过滤 —— 用滚动方差比判断"当前是不是趋势市", 只在趋势市里做交易。

运行:  python my_try/10_regime_filter.py      (只读数据, 不做回测)

思路来源(前几步的结论):
    08 热力图  -> 参数在 fast 8~12 / slow 25~40 是一片高原, 周期选择不是瓶颈
    09 方差比  -> VR 在 q≈5~50 > 1 (弱趋势), q≥80 < 1 (弱均值回归)
    08 逐年    -> 2025 年 8 组参数全亏; 而 09 显示那年 VR(40)=0.90 (均值回归)
    -> 所以趋势策略的失效是"整年"级别的, 不是"某个参数"级别的。
       有价值的自适应应该是一个"开关"(该不该做趋势), 而不是"旋钮"(用哪个周期)。

这个脚本做两件事:

  第一部分 可预测性检验
      对每个时刻 t, 只用 [t-W, t] 的数据算滚动 VR; 再看"接下来 H 小时"的
      趋势策略收益。如果 VR 高的时候未来收益确实更好, 说明 regime 可预测。

  第二部分 把 regime 过滤当成真实交易规则
      与原策略在同一个 4.75 年上对比: 总收益、逐年收益、空仓时间占比。
      严格无前视偏差: 判断 regime 只用过去数据。

注意: 成交仍是简化假设(收盘价反手、固定 1 手、每手固定手续费)。
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
SYMBOL = "KQ.i@SHFE.rb"
DURATION = 60 * 60
DATA_LEN = 8000
VOLUME_MULTIPLE = 10
FEE_PER_LOT = 3.0
STOP_LOSS_PCT = 0.02
INIT_CAPITAL = 100_000

FAST, SLOW = 10, 30          # 基础策略参数(取热力图高原中心)
VR_Q = 40                    # 用 40 小时算方差比
VR_WIN = 500                 # 滚动窗口长度(小时), 约 3 个月
VR_STEP = 5                  # 每 5 根K线重算一次 VR(实盘也不会每根都重算)
FWD_H = 40                   # 可预测性检验里"未来 H 小时"
VR_THRESHOLDS = [1.00, 1.02, 1.05, 1.08]    # 候选门槛


def vr_stat(logp, q):
    """Lo-MacKinlay 方差比: 返回 (VR(q), z)。"""
    r = np.diff(logp)
    T = len(r)
    if T <= q + 1:
        return float("nan"), float("nan")
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / (T - 1)
    rq = logp[q:] - logp[:-q]
    m = q * (T - q + 1) * (1 - q / T)
    varq = np.sum((rq - q * mu) ** 2) / m
    vr = varq / var1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    return float(vr), float(z)


def rolling_vr(logp, q, win, step):
    """滚动 VR(q): out[t] 只使用 logp[t-win : t+1] 的数据(无前视偏差)。
       每 step 根K线重算一次, 其余时刻沿用上一次的值 —— 这也是实盘的实际情况。"""
    n = len(logp)
    out = np.full(n, np.nan)
    last = np.nan
    for t in range(win, n):
        if (t - win) % step == 0:
            last = vr_stat(logp[t - win:t + 1], q)[0]
        out[t] = last
    return out


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if len(equity) else 0.0


def simulate_regime(close, vr, fast, slow, thr=float("nan"),
                    fee=FEE_PER_LOT, mult=VOLUME_MULTIPLE, stop_pct=STOP_LOSS_PCT):
    """与 08 相同的双均线反手 + 止损, 额外加一层 regime 开关。

    thr 为 nan  -> 不过滤(原始策略)
    thr 为数值 -> 仅当 vr[i] > thr 时允许持仓; 一旦 VR 掉下来就平仓并停止开新仓
    返回 (逐根盈亏, 统计字典)
    """
    n = len(close)
    px = np.asarray(close, dtype=float)
    ma_f = pd.Series(px).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(px).rolling(slow).mean().to_numpy()

    pnl = np.zeros(n)
    pos, entry = 0, np.nan
    stopped = False
    turnover = 0.0
    reversals = 0
    blocked_bars = 0        # 因 regime 不合格被禁止交易的K线数

    use_filter = thr == thr     # thr 不是 nan
    for i in range(1, n):
        if pos != 0:
            pnl[i] += pos * (px[i] - px[i - 1]) * mult

        f, s, fp, sp = ma_f[i], ma_s[i], ma_f[i - 1], ma_s[i - 1]
        if not (f == f and s == s and fp == fp and sp == sp):
            continue

        golden = fp <= sp and f > s
        dead = fp >= sp and f < s
        target = pos
        if golden:
            stopped, target = False, 1
        elif dead:
            stopped, target = False, -1

        if not stopped and pos != 0 and entry == entry:
            ret = (px[i] - entry) / entry * pos
            if ret <= -stop_pct:
                stopped, target = True, 0
        if stopped:
            target = 0

        # ---- regime 开关: 判断只用过去数据 ----
        if use_filter:
            ok = (vr[i] == vr[i]) and vr[i] > thr
            if not ok:
                target = 0
                blocked_bars += 1

        if target != pos:
            lots = abs(target - pos)
            turnover += lots
            pnl[i] -= lots * fee
            if pos != 0 and target != 0:
                reversals += 1
            pos = target
            entry = px[i] if pos != 0 else np.nan

    stat = {"turnover": turnover, "reversals": reversals,
            "fee": turnover * fee, "blocked": blocked_bars,
            "blocked_pct": blocked_bars / (n - 1) * 100}
    return pnl, stat


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

    if len(df) < 2000:
        raise SystemExit(f"拿到的数据太少({len(df)} 根), 请检查网络后重跑。")

    df["dt"] = pd.to_datetime(df["datetime"], unit="ns") + pd.Timedelta(hours=8)
    close = df["close"].to_numpy(dtype=float)
    logp = np.log(close)
    years = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days / 365.25

    print()
    print("=" * 100)
    print(f"Regime 过滤   标的 {SYMBOL}   周期 {DURATION // 60} 分钟")
    print(f"区间 {df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d}  "
          f"{len(df)} 根 / {years:.2f} 年")
    print(f"基础策略 {FAST}/{SLOW}   滚动 VR(q={VR_Q}, 窗口={VR_WIN}h, 每 {VR_STEP} 根重算)")
    print("=" * 100)

    # ================= 第一部分: 可预测性检验 =================
    vr = rolling_vr(logp, VR_Q, VR_WIN, VR_STEP)
    ma_f = pd.Series(close).rolling(FAST).mean().to_numpy()
    ma_s = pd.Series(close).rolling(SLOW).mean().to_numpy()
    pos = np.sign(ma_f - ma_s)
    fwd = np.full(len(close), np.nan)
    fwd[:-FWD_H] = pos[:-FWD_H] * (close[FWD_H:] - close[:-FWD_H]) / close[:-FWD_H]

    ok = ~np.isnan(vr) & ~np.isnan(fwd)
    v, y = vr[ok], fwd[ok]
    r, pval = pearsonr(v, y)

    print()
    print("【第一部分】可预测性检验 —— 滚动 VR 能不能预测接下来的收益?")
    print("-" * 100)
    print(f"样本 {len(v)} 个时刻; 检验: corr( 滚动VR , 未来 {FWD_H} 小时趋势策略收益 )")
    g = pd.qcut(pd.Series(v), 5, labels=False, duplicates="drop")
    tbl = pd.DataFrame({"vr": v, "fwd": y, "g": g})
    print(f"\n{'VR 分组':<10}{'样本数':>8}{'VR 区间':>20}{'未来收益均值':>14}{'胜率':>10}")
    print("-" * 100)
    for gi, part in tbl.groupby("g"):
        print(f"{'第' + str(int(gi) + 1) + '组':<10}{len(part):>8}"
              f"{f'{part.vr.min():.3f} ~ {part.vr.max():.3f}':>20}"
              f"{part.fwd.mean() * 100:>13.4f}%{(part.fwd > 0).mean() * 100:>9.1f}%")
    print()
    print(f"相关系数 r = {r:+.4f}   p 值 = {pval:.4f}   "
          f"({'显著' if pval < 0.05 else '不显著'}, 5% 水平)")
    print(f"判读: r 显著为正 -> VR 高时未来趋势收益更好, regime 可预测;")
    print(f"      不显著     -> VR 只能事后解释, 不能事前预测, 过滤没用。")

    # ================= 第二部分: 当成真实交易规则 =================
    print()
    print("【第二部分】把 regime 过滤当成交易规则 —— 与不过滤的原策略对比")
    print("-" * 100)
    print(f"{'门槛':<12}{'总收益率':>10}{'年化':>9}{'最大回撤':>10}{'夏普':>8}"
          f"{'反手':>7}{'空仓占比':>10}{'手续费':>10}")
    print("-" * 100)

    results = {}
    order = [float("nan")] + VR_THRESHOLDS
    bp_year = len(df) / years
    for thr in order:
        pnl, st = simulate_regime(close, vr, FAST, SLOW, thr=thr)
        eq = np.cumsum(pnl)
        total = float(pnl.sum())
        rr = pnl / INIT_CAPITAL
        shp = float(rr.mean() / rr.std() * np.sqrt(bp_year)) if rr.std() > 0 else float("nan")
        label = "不过滤" if thr != thr else f"VR > {thr:.2f}"
        print(f"{label:<12}{total / INIT_CAPITAL * 100:>9.1f}%"
              f"{total / INIT_CAPITAL / years * 100:>8.1f}%"
              f"{max_drawdown(eq) / INIT_CAPITAL * 100:>9.1f}%{shp:>8.2f}"
              f"{st['reversals']:>7}{st['blocked_pct']:>9.1f}%{st['fee']:>10,.0f}")
        results[label] = pd.Series(pnl).groupby(df["dt"].dt.year).sum() / INIT_CAPITAL * 100

    print()
    print("逐年收益率(%):")
    ydf = pd.DataFrame(results).T
    ydf = ydf.reindex(columns=sorted(ydf.columns))
    print(f"{'门槛':<12}" + "".join(f"{c:>10}" for c in ydf.columns) + f"{'合计':>10}")
    print("-" * 100)
    for label in ydf.index:
        print(f"{label:<12}" + "".join(f"{v:>9.1f}%" for v in ydf.loc[label])
              + f"{ydf.loc[label].sum():>9.1f}%")


if __name__ == "__main__":
    main()
