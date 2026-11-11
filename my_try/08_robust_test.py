# -*- coding: utf-8 -*-
"""
稳健性检验 —— 用 4.75 年历史, 一次扫完几十组参数。

运行:  python my_try/08_robust_test.py       (只读数据, 不做回测)

为什么需要这个脚本:
  04_strategy.py 用 TqApi 回测, 但单个期货合约的生命周期只有约 12 个月
  (上市 -> 交割), 所以最多只能回测几个月到一年。要判断"20/60 的收益是
  真实规律还是运气", 必须看更长的时间跨度。

做法:
  * 用指数合约 KQ.i@SHFE.rb 的 1 小时线(最多 8000 根 ≈ 4.75 年)。
    指数合约按各月份合约加权平均, 换月时权重渐变, 价格连续, 没有主连那种
    换月跳空, 是长历史信号研究的正确原料。
  * 指数合约不能直接交易, 所以这里的成交是"简化假设": 信号出现就按
    K线收盘价反手, 固定 1 手, 按每手固定金额扣手续费。
  * 因此: 绝对收益不能和 04_strategy.py 的结果直接对比;
    但**参数之间的相对排序是可信的**, 而排序的稳定性正是我们要检验的东西。

止损逻辑、金叉死叉的判定都与 04_strategy.py 保持一致。
"""
import time

import numpy as np
import pandas as pd
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
SYMBOL = "KQ.i@SHFE.rb"         # 指数合约: 连续、无换月跳空、历史长
DURATION = 60 * 60              # 1 小时线
DATA_LEN = 8000                 # 最多 8000 根 ≈ 4.75 年
VOLUME_MULTIPLE = 10            # 螺纹合约乘数(10 吨/手)
FEE_PER_LOT = 3.0               # 手续费 元/手(真实量级); TqSim 默认只有约 0.31
STOP_LOSS_PCT = 0.02            # 单笔止损: 相对开仓价逆向 2%
INIT_CAPITAL = 100_000          # 名义初始资金, 只用来把盈亏换算成收益率
PARAM_SETS = [(5, 15), (5, 20), (10, 30), (10, 40),
              (15, 45), (20, 40), (20, 60), (30, 90)]

# 热力图网格(用来判断"参数高原"还是"参数孤岛")
RUN_HEATMAP = True
HEAT_FAST = [5, 8, 10, 12, 15, 20, 25, 30]
HEAT_SLOW = [20, 25, 30, 35, 40, 50, 60, 80, 100]

# 手续费敏感性: 同一批策略放在这些费率下重跑(元/手)
RUN_COST_SENS = True
FEE_LEVELS = [0.0, 0.31, 1.0, 3.0, 6.0, 10.0, 20.0]
REAL_FEE_RANGE = (1.0, 3.0)     # 真实螺纹手续费的大致区间(元/手), 用来做可行性判断


def simulate(close, fast, slow, fee=FEE_PER_LOT, mult=VOLUME_MULTIPLE, stop_pct=STOP_LOSS_PCT):
    """逐根K线模拟双均线反手 + 止损。返回 (逐根盈亏, 统计字典)。

    与 04_strategy.py 的规则一致:
      * 金叉 -> 目标 +1 手; 死叉 -> 目标 -1 手
      * 浮动亏损达到 stop_pct -> 平仓并把 stopped 置真, 等下一次交叉才重新进场
      * 每根K线的盈亏在**下一根**才体现(持仓在本根收盘时才调整), 避免未来函数
    """
    n = len(close)
    px = np.asarray(close, dtype=float)
    ma_f = pd.Series(px).rolling(fast).mean().to_numpy()
    ma_s = pd.Series(px).rolling(slow).mean().to_numpy()

    pnl = np.zeros(n)
    pos, entry = 0, np.nan
    stopped = False
    turnover = 0.0      # 累计换手手数
    reversals = 0       # 反手次数(多<->空)
    stops = 0           # 止损次数

    for i in range(1, n):
        # 1) 本根K线的浮动盈亏(用上一根收盘时的持仓)
        if pos != 0:
            pnl[i] += pos * (px[i] - px[i - 1]) * mult

        f, s, fp, sp = ma_f[i], ma_s[i], ma_f[i - 1], ma_s[i - 1]
        if not (f == f and s == s and fp == fp and sp == sp):
            continue        # 均线还没预热完

        # 2) 信号层
        golden = fp <= sp and f > s
        dead = fp >= sp and f < s
        target = pos
        if golden:
            stopped, target = False, 1
        elif dead:
            stopped, target = False, -1

        # 3) 风控层: 止损优先级高于信号
        if not stopped and pos != 0 and entry == entry:
            ret = (px[i] - entry) / entry * pos      # 按持仓方向的收益率
            if ret <= -stop_pct:
                stopped, target = True, 0
                stops += 1
        if stopped:
            target = 0

        # 4) 执行层: 换手 + 手续费
        if target != pos:
            lots = abs(target - pos)
            turnover += lots
            pnl[i] -= lots * fee
            if pos != 0 and target != 0:
                reversals += 1
            pos = target
            entry = px[i] if pos != 0 else np.nan

    stat = {"turnover": turnover, "reversals": reversals, "stops": stops,
            "fee": turnover * fee}
    return pnl, stat


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    return float(np.max(peak - equity)) if len(equity) else 0.0


def param_heatmap(close, df):
    """扫描 fast × slow 网格, 打印总收益率矩阵, 判断"参数高原"还是"参数孤岛"。

    这是判断"要不要继续折腾参数"的第一道门:
      * 高原 —— 最优参数周围一片都不错 -> 策略对参数不敏感, 有真实机制,
                那就没必要去追"最优值"
      * 孤岛 —— 只有最优参数一个好, 邻居全差 -> 那个收益是碰运气, 不可信
    """
    years = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days / 365.25
    grid = pd.DataFrame(index=HEAT_FAST, columns=HEAT_SLOW, dtype=float)
    for fast in HEAT_FAST:
        for slow in HEAT_SLOW:
            if fast >= slow:          # 快线必须短于慢线
                continue
            pnl, _ = simulate(close, fast, slow)
            grid.loc[fast, slow] = float(pnl.sum()) / INIT_CAPITAL * 100

    print()
    print("=" * (10 + 10 * len(HEAT_SLOW)))
    print("参数热力图 —— 总收益率(%), 固定 1 手 / 名义资金 100,000")
    print("行 = fast(快线周期), 列 = slow(慢线周期), '--' 表示该组合无效")
    print(f"区间 {df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d}  ({years:.2f} 年)")
    print("=" * (10 + 10 * len(HEAT_SLOW)))
    print(f"{'fast/slow':<10}" + "".join(f"{s:>10}" for s in HEAT_SLOW))
    print("-" * (10 + 10 * len(HEAT_SLOW)))
    for fast in HEAT_FAST:
        line = f"{fast:<10}"
        for slow in HEAT_SLOW:
            v = grid.loc[fast, slow]
            line += f"{'--':>10}" if v != v else f"{v:>9.1f}%"
        print(line)

    # ---------- 高原 / 孤岛 的量化判断 ----------
    flat = grid.stack()
    n_all, n_pos = len(flat), int((flat > 0).sum())
    best_f, best_s = flat.idxmax()
    bi, bj = HEAT_FAST.index(best_f), HEAT_SLOW.index(best_s)
    nb = grid.iloc[max(bi - 1, 0):bi + 2, max(bj - 1, 0):bj + 2].stack()

    print()
    print(f"有效组合 {n_all} 个, 其中正收益 {n_pos} 个 ({n_pos / n_all * 100:.0f}%)")
    print(f"全局最优: fast={best_f} slow={best_s} -> {flat.max():.1f}%")
    print(f"该点 3x3 邻域: 均值 {nb.mean():.1f}%   最好 {nb.max():.1f}%   最差 {nb.min():.1f}%")
    print()
    print("判读标准:")
    print("  邻域均值 接近 最优值   -> 参数高原: 对参数不敏感, 结论可信, 不必追'最优值'")
    print("  邻域均值 远低于 最优值 -> 参数孤岛: 收益靠运气, 换个参数就崩, 不可信")
    print()
    print("收益率前 5 名(看它们是否聚集在一起):")
    for (f, s), v in flat.sort_values(ascending=False).head(5).items():
        print(f"  fast={f:<3} slow={s:<5} {v:>7.1f}%")


def cost_sensitivity(close, df):
    """把同一批参数放在不同手续费下重跑, 看利润空间到底有多薄。

    核心指标 —— 平衡手续费 = 毛利 / 换手手数:
        即"手续费涨到多少元/手, 这个策略才开始亏钱"。
        毛利(手续费=0 时的盈亏)只取决于行情, 换手手数只取决于信号,
        两者都与手续费无关, 所以平衡手续费是个干净的"成本承受力"指标。
    """
    years = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days / 365.25
    w = 9
    total_w = 34 + w * len(FEE_LEVELS)

    rows = []
    for fast, slow in PARAM_SETS:
        res, stat0 = {}, None
        for f in FEE_LEVELS:
            pnl, stat = simulate(close, fast, slow, fee=f)
            res[f] = float(pnl.sum())
            if f == 0.0:
                stat0 = stat
        turnover = stat0["turnover"]
        rows.append({
            "label": f"{fast}/{slow}", "reversals": stat0["reversals"],
            "gross": res[0.0], "turnover": turnover,
            "be_fee": res[0.0] / turnover if turnover > 0 else float("nan"),
            "res": res,
        })

    print()
    print("=" * total_w)
    print("手续费敏感性 —— 同一批策略在不同手续费下的总收益率(%)")
    print("固定 1 手 / 名义资金 100,000"
          f"   区间 {df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d} ({years:.2f} 年)")
    print("=" * total_w)
    print(f"{'参数':<8}{'反手':>6}{'毛利(元)':>11}{'平衡手续费':>12}"
          + "".join(f"{f:>{w}g}" for f in FEE_LEVELS))
    print("-" * total_w)
    for r in rows:
        cells = "".join(f"{r['res'][f] / INIT_CAPITAL * 100:>{w - 1}.1f}%" for f in FEE_LEVELS)
        print(f"{r['label']:<8}{r['reversals']:>6}{r['gross']:>11,.0f}"
              f"{r['be_fee']:>12.1f}{cells}")

    lo, hi = REAL_FEE_RANGE
    ok = [r["label"] for r in rows if r["be_fee"] >= hi]
    marginal = [r["label"] for r in rows if lo <= r["be_fee"] < hi]
    dead = [r["label"] for r in rows if r["be_fee"] < lo]

    print()
    print("判读:")
    print("  平衡手续费 = 毛利 ÷ 换手手数 = '手续费涨到多少元/手才开始亏钱'。")
    print(f"  真实螺纹手续费大约 {lo:g}~{hi:g} 元/手; TqSim 默认只有 0.31 元/手, 偏低了约一个量级。")
    print()
    print(f"  成本空间充足 (平衡点 ≥ {hi:g} 元/手): {ok if ok else '无'}")
    print(f"  勉强可行   (平衡点 {lo:g}~{hi:g} 元/手): {marginal if marginal else '无'}")
    print(f"  真实成本下不可行 (平衡点 < {lo:g} 元/手): {dead if dead else '无'}")


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

    if len(df) < 500:
        raise SystemExit(f"拿到的数据太少({len(df)} 根), 请检查网络后重跑。")

    df["dt"] = pd.to_datetime(df["datetime"], unit="ns") + pd.Timedelta(hours=8)
    close = df["close"]
    years = (df["dt"].iloc[-1] - df["dt"].iloc[0]).days / 365.25
    bars_per_year = len(df) / years

    print()
    print("=" * 96)
    print(f"标的 {SYMBOL} (指数合约, 无换月跳空)   周期 {DURATION // 60} 分钟")
    print(f"区间 {df['dt'].iloc[0]:%Y-%m-%d} ~ {df['dt'].iloc[-1]:%Y-%m-%d}   "
          f"{len(df)} 根K线 / {years:.2f} 年")
    print(f"手续费 {FEE_PER_LOT} 元/手   合约乘数 {VOLUME_MULTIPLE}   "
          f"名义资金 {INIT_CAPITAL:,}   止损 {STOP_LOSS_PCT * 100:.1f}%")
    print("=" * 96)
    print(f"{'参数':<8}{'总盈亏(元)':>12}{'总收益率':>10}{'年化':>9}"
          f"{'最大回撤':>10}{'夏普':>8}{'反手次数':>9}{'手续费(元)':>11}{'手续费/总盈亏':>13}")
    print("-" * 96)

    rows, yearly = [], {}
    for fast, slow in PARAM_SETS:
        pnl, stat = simulate(close, fast, slow)
        equity = np.cumsum(pnl)
        total = float(pnl.sum())
        ror = total / INIT_CAPITAL
        ann = ror / years if years > 0 else float("nan")
        mdd = max_drawdown(equity)
        r = pnl / INIT_CAPITAL
        sharpe = float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else float("nan")
        fee_ratio = stat["fee"] / total if total > 0 else float("nan")

        label = f"{fast}/{slow}"
        print(f"{label:<8}{total:>12,.0f}{ror * 100:>9.1f}%{ann * 100:>8.1f}%"
              f"{mdd / INIT_CAPITAL * 100:>9.1f}%{sharpe:>8.2f}{stat['reversals']:>9}"
              f"{stat['fee']:>11,.0f}{fee_ratio * 100:>12.1f}%")

        rows.append((label, total, ror, ann, mdd, sharpe, stat))
        yearly[label] = pd.Series(pnl).groupby(df["dt"].dt.year).sum() / INIT_CAPITAL * 100

    # ---------- 逐年收益率矩阵(核心输出) ----------
    ydf = pd.DataFrame(yearly).T          # 行=参数, 列=年份
    ydf = ydf.reindex(columns=sorted(ydf.columns))
    print()
    print("=" * 96)
    print("逐年收益率矩阵(%)  —— 看参数排序是否跨年稳定")
    print("=" * 96)
    print(f"{'参数':<8}" + "".join(f"{c:>10}" for c in ydf.columns) + f"{'盈利年数':>10}")
    print("-" * 96)
    for label in ydf.index:
        vals = ydf.loc[label]
        full = vals.dropna()
        print(f"{label:<8}" + "".join(
            f"{v:>9.1f}%" if v == v else f"{'--':>10}" for v in vals) +
            f"{int((full > 0).sum())}/{len(full):>3}")

    # ---------- 稳定性的量化判断 ----------
    print()
    print("=" * 96)
    print("判读提示")
    print("=" * 96)
    std_of_ranks = ydf.rank(axis=0, ascending=False).std(axis=1)
    print(f"{'参数':<8}{'平均名次':>10}{'名次波动(标准差)':>18}{'总收益率':>12}")
    print("-" * 96)
    ranks = ydf.rank(axis=0, ascending=False)
    for label in ydf.index:
        print(f"{label:<8}{ranks.loc[label].mean():>10.2f}"
              f"{std_of_ranks[label]:>18.2f}{ydf.loc[label].sum():>11.1f}%")
    print()
    print("名次波动越小 -> 该参数在各种年份里表现越稳定;")
    print("若各参数的名次逐年剧烈变化, 说明'哪个参数最好'是随行情漂移的, 没有稳定规律。")

    if RUN_HEATMAP:
        param_heatmap(close, df)

    if RUN_COST_SENS:
        cost_sensitivity(close, df)


if __name__ == "__main__":
    main()
