# -*- coding: utf-8 -*-
"""
多品种组合回测 —— TSMOM(时序动量) + carry(展期收益), 目标: 为一个
「11.11 ~ 12.15, 约 24 个交易日, 按收益率排名」的模拟盘考核选出可下注的组合。

运行:  python my_try/20_portfolio_backtest.py       (只读数据, 不下单)

为什么换成组合:
    08~10 证明【单品种】螺纹的趋势策略期望≈0。但文献里商品期货最稳健的
    正期望来源是【多品种时序动量】(Moskowitz-Ooi-Pedersen, JFE 2012):
    单品种信号弱且噪声大, 30 个近似独立的弱信号叠加后, 组合层面的
    夏普显著上升, 这正是 09 号 VR(5~50)>1 在组合上才兑现的地方。

口径(与 12/14 号一致, 不重新发明):
  * 换月表来自 https://files.shinnytech.com/continuous_table.json
  * 收益 = 可交易连续(ratio-adjusted): 换月日用新合约自己的日收益, 段内用
    同一合约日收益, 复利累积。绝不用主连 KQ.m@(换月跳空是假象)。
  * 数据取 output/cache_14_daily.pkl (14 号跑的合约级日线缓存, 无需重新联网)

回测规则(尽量贴近一个真实可执行的学生模拟盘):
  1. 每周最后一个交易日收盘后调仓一次(24 个交易日 ≈ 5 次调仓)
  2. 每个品种独立给定风险预算, 仓位 = 信号方向 × 风险预算 / 该品种波动率
  3. 单品种仓位上限 + 组合总杠杆上限, 避免个别品种吃掉全部风险
  4. 按换手扣成本, 并做 3 / 6 / 12 bp 敏感性
  5. 关键输出: 全期绩效 + 【24 个交易日滚动窗口】的收益分布
     —— 后者才是"这次考核可能考成什么样"的直接答案
"""
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "output" / "cache_14_daily.pkl"
TABLE_CACHE = HERE / "output" / "continuous_table.json"
CONT_TABLE_URL = "https://files.shinnytech.com/continuous_table.json"
EX_OK = {"SHFE", "DCE", "CZCE", "INE", "GFEX"}

SIG_FAST = 20          # 快信号回看天数
SIG_SLOW = 60          # 慢信号回看天数
VOL_WIN = 60           # 波动率估计窗口(日)
MIN_HIST = 120         # 品种至少要有多少根日线才纳入
VOL_FLOOR = 0.08       # 年化波动率下限, 防止低波动品种仓位爆炸
LEVERAGE = float(os.environ.get("BT_LEVERAGE", "2.0"))  # 总风险预算, 用环境变量做杠杆刻度
PER_INST_RISK = 0.10   # 单品种名义仓位(占权益)上限倍数
COST_BPS_SET = [3.0, 6.0, 12.0]
WINDOW = 24            # 考核期交易日数
N_LONG_WIN = 8         # 滚动窗口至少需要多少天
# 混合档: 70% 稳腿(全品种 carry 多空) + 30% 搏腿(做多 carry 最强的 N 个品种, 放大 SAT_LEV 倍)
CORE_FRAC = 0.70
SAT_N = 3
SAT_LEV = 3.0


def get_rollovers(table, key):
    rows = table.get(key)
    if not rows:
        return None
    out = [(datetime.strptime(d, "%Y%m%d").date(), sym) for d, sym in rows]
    out.sort(key=lambda x: x[0])
    return out


def contract_month(sym):
    code = sym.split(".")[-1]
    digits = ""
    for ch in reversed(code):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if len(digits) == 4:
        y, m = 2000 + int(digits[:2]), int(digits[2:])
    elif len(digits) == 3:
        y, m = 2020 + int(digits[:1]), int(digits[1:])
    else:
        return None
    return y * 12 + m if 1 <= m <= 12 else None


def pad_cn(text, width):
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - w)


def load_table():
    if TABLE_CACHE.exists():
        import json
        return json.loads(TABLE_CACHE.read_text(encoding="utf-8"))
    import requests
    print("下载换月表 ...")
    table = requests.get(CONT_TABLE_URL, timeout=30).json()
    TABLE_CACHE.parent.mkdir(exist_ok=True)
    import json
    TABLE_CACHE.write_text(json.dumps(table), encoding="utf-8")
    return table


def build_returns(table, close):
    """每个品种 -> 可交易连续日收益 Series(ratio-adjusted)。"""
    varieties = sorted({k for k in table if k.split(".")[0] in EX_OK})
    rets, ret_dates = {}, {}
    for k in varieties:
        roll = get_rollovers(table, k)
        if not roll or len(roll) < 3:
            continue
        # 每个交易日 -> 当期主力合约
        main_by_date = {}
        for i in range(len(roll) - 1):
            s, e, sym = roll[i][0], roll[i + 1][0], roll[i][1]
            sn = close.get(sym)
            if sn is None:
                continue
            for d in sn.index:
                if s <= d.date() < e:
                    main_by_date[d] = sym
        days = sorted(main_by_date)
        if len(days) < MIN_HIST + VOL_WIN:
            continue
        dts, rs = [], []
        for j in range(1, len(days)):
            d, dp = days[j], days[j - 1]
            sn = close.get(main_by_date[d])
            if sn is None or d not in sn.index or dp not in sn.index:
                continue
            v0, v1 = float(sn.loc[dp]), float(sn.loc[d])
            if v0 <= 0:
                continue
            dts.append(d)
            rs.append(v1 / v0 - 1.0)
        if len(rs) < MIN_HIST:
            continue
        rets[k] = pd.Series(rs, index=pd.DatetimeIndex(dts))
        ret_dates[k] = set(dts)
    return rets


def build_carry(table, close, calendar):
    """每个品种在每个交易日 -> 年化展期收益 (P近/P远-1)/(月差/12)。"""
    out = {}
    varieties = sorted({k for k in table if k.split(".")[0] in EX_OK})
    for k in varieties:
        roll = get_rollovers(table, k)
        if not roll or len(roll) < 3:
            continue
        vals = {}
        for i in range(len(roll) - 1):
            near, far = roll[i][1], roll[i + 1][1]
            sn, sf = close.get(near), close.get(far)
            if sn is None or sf is None:
                continue
            gap = contract_month(far) - contract_month(near)
            if not gap or not (1 <= gap <= 18):
                continue
            both = sn.index.intersection(sf.index)
            for d in both:
                if d not in calendar:
                    continue
                pn, pf = float(sn.loc[d]), float(sf.loc[d])
                if pn > 0 and pf > 0:
                    vals[d] = (pn / pf - 1.0) / (gap / 12.0)
        if vals:
            out[k] = pd.Series(vals).sort_index()
    return out


def perf(nav, ret, label, cost_bps):
    """年化收益 / 波动 / 夏普 / 最大回撤。"""
    n = len(nav)
    if n < 30:
        return None
    yrs = n / 252.0
    total = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (1 + total) ** (1 / yrs) - 1
    vol = ret.std(ddof=1) * np.sqrt(252)
    sharpe = (ret.mean() * 252 - 0.02) / vol if vol > 0 else np.nan
    dd = (nav / nav.cummax() - 1).min()
    return dict(label=label, cost=cost_bps, cagr=cagr, vol=vol, sharpe=sharpe,
                mdd=dd, total=total, n=n)


def main():
    print("=" * 100)
    print("多品种组合回测 · TSMOM + carry · 目标: 24 个交易日排名考核")
    print("=" * 100)

    if not CACHE.exists():
        print(f"缺少缓存 {CACHE}, 请先运行 14_cross_section.py")
        return 1
    cache = pickle.load(open(CACHE, "rb"))
    close, volu = cache["close"], cache["volu"]
    table = load_table()

    rets = build_returns(table, close)
    if not rets:
        print("没有可用品种")
        return 1

    # ---------- 主日历: 取各品种交易日的并集, 保留多数品种有数据的日子 ----------
    all_dates = sorted(set().union(*[set(s.index) for s in rets.values()]))
    cal = pd.DatetimeIndex(all_dates)
    R = pd.DataFrame({k: s.reindex(cal) for k, s in rets.items()})
    cover = R.notna().sum(axis=1) / R.shape[1]
    cal = cal[cover >= 0.5]
    R = R.loc[cal]
    print(f"品种数 {R.shape[1]}, 交易日 {len(cal)} 天 "
          f"({cal[0].date()} ~ {cal[-1].date()}, {len(cal)/252:.2f} 年)")

    # ---------- 信号 ----------
    idx = (1 + R.fillna(0)).cumprod()        # 可交易连续指数(用于算信号)
    idx = idx.where(R.notna().cummax(), np.nan)
    sigma = R.rolling(VOL_WIN, min_periods=VOL_WIN // 2).std() * np.sqrt(252)
    sigma = sigma.clip(lower=VOL_FLOOR)

    mom = (0.5 * (idx / idx.shift(SIG_FAST) - 1.0)
           + 0.5 * (idx / idx.shift(SIG_SLOW) - 1.0))
    mom_z = mom.div(sigma)                   # 风险调整后的动量强度

    carry = build_carry(table, close, set(cal))
    C = pd.DataFrame({k: s.reindex(cal) for k, s in carry.items()})
    C = C.reindex(columns=R.columns)
    print(f"carry 覆盖: 平均 {C.notna().sum(axis=1).mean():.1f} 个品种/日, "
          f"有效日 {int((C.notna().sum(axis=1) > 5).sum())}/{len(cal)}")

    # ---------- 调仓日: 每周最后一个交易日 ----------
    iso = cal.isocalendar()
    week = pd.Series(cal).groupby([iso["year"].values, iso["week"].values]).max()
    rebal = pd.DatetimeIndex(sorted(week.values))
    rebal = rebal[rebal >= cal[VOL_WIN]]
    print(f"调仓次数 {len(rebal)} 次 (每周一次)")

    # ---------- 组合权重 ----------
    # 每个品种的原始风险权重: 信号方向 x 1/波动率, 再归一化到固定总风险
    def raw_weights(t, kind):
        m = mom_z.loc[t].copy() if kind in ("mom", "mom_rev", "combo") else pd.Series(np.nan, index=R.columns)
        c = np.nan
        if kind in ("carry", "combo", "mix"):
            cv = C.loc[t].dropna()
            if len(cv) >= 8:
                rk = cv.rank(pct=True) - 0.5        # 横截面: 高 carry 做多, 低 carry 做空
                c = (rk / 0.5).reindex(R.columns)
        if kind == "mix":
            # 稳腿(70%): 全品种多空 carry, 与 CARRY 同口径但只占 70% 风险
            wc = raw_weights(t, "carry") * CORE_FRAC
            # 搏腿(30%): 【同一个 carry 信号】但只取最极端的多/空各 N 个品种, 放大 SAT_LEV 倍。
            # 关键: 仍是多空对冲, 不偷偷加商品 beta —— 只放大方差, 不改变期望来源。
            cv = C.loc[t].dropna()
            ws = pd.Series(0.0, index=R.columns)
            if len(cv) >= 2 * SAT_N:
                per = SAT_LEV * (1 - CORE_FRAC) / SAT_N
                ws[cv.nlargest(SAT_N).index] = per
                ws[cv.nsmallest(SAT_N).index] = -per
            return (wc + ws).reindex(R.columns).fillna(0.0)
        if kind == "mom":
            score = np.sign(m)
        elif kind == "mom_rev":
            score = -np.sign(m)                     # 反向: 做空过去的赢家
        elif kind == "carry":
            score = np.sign(c)
        else:
            with np.errstate(invalid="ignore"):
                score = np.nanmean(np.vstack([np.sign(m).values, np.sign(c).values]), axis=0)
            score = pd.Series(score, index=R.columns)
        w = score / sigma.loc[t]
        w = w.replace([np.inf, -np.inf], np.nan).dropna()
        if w.empty:
            return pd.Series(0.0, index=R.columns)
        gross = w.abs().sum()
        w = w / gross * LEVERAGE           # 总风险预算 = LEVERAGE(与 15% 年化波动同量级)
        return w.clip(-PER_INST_RISK, PER_INST_RISK).reindex(R.columns).fillna(0.0)

    # ---------- 表 0: 信号是否真的有预测力 (IC) ----------
    print("\n" + "=" * 100)
    print("表 0  信号 IC (横截面 Spearman, 每个调仓日算一次再取均值)")
    print("      IC>0 = 信号方向对; 显著为负 = 应反向使用; ≈0 = 信号无用")
    print("=" * 100)
    fwd_5 = idx.shift(-5) / idx - 1.0
    fwd_1 = idx.shift(-1) / idx - 1.0
    print(f"{'信号':<14}{'预测1日':>12}{'预测5日':>12}{'标准差':>10}{'t值(5日)':>12}")
    print("-" * 100)
    for name, sig in [("mom_z(20/60)", mom_z), ("carry(年化展期)", C)]:
        ics = []
        for t in rebal:
            s, f = sig.loc[t].dropna(), fwd_5.loc[t]
            ok = s.index.intersection(f.dropna().index)
            if len(ok) < 8 or s[ok].nunique() < 3:
                continue
            ics.append(s[ok].rank().corr(f[ok].rank()))
        ics = [x for x in ics if not np.isnan(x)]
        if not ics:
            continue
        ic1 = []
        for t in rebal:
            s, f = sig.loc[t].dropna(), fwd_1.loc[t]
            ok = s.index.intersection(f.dropna().index)
            if len(ok) >= 8 and s[ok].nunique() >= 3:
                ic1.append(s[ok].rank().corr(f[ok].rank()))
        ic1 = [x for x in ic1 if not np.isnan(x)]
        m5, sd5, n5 = np.mean(ics), np.std(ics, ddof=1), len(ics)
        t5 = m5 / (sd5 / np.sqrt(n5)) if sd5 > 0 else np.nan
        print(f"{pad_cn(name, 14)}{np.mean(ic1):>12.3f}{m5:>12.3f}{sd5:>10.3f}{t5:>12.2f}")

    results = {}
    for kind, label in [("mom", "TSMOM"), ("mom_rev", "TSMOM-REV"), ("carry", "CARRY"),
                        ("mix", "MIX70/30"),
                        ("combo", "TSMOM+CARRY"), ("long", "EQUAL-LONG")]:
        W = pd.DataFrame(0.0, index=rebal, columns=R.columns)
        for t in rebal:
            if kind == "long":
                W.loc[t] = LEVERAGE / max(1, R.loc[t].notna().sum())
            else:
                W.loc[t] = raw_weights(t, kind)
        # 权重持有到下一个调仓日
        pos = W.reindex(cal, method="ffill").shift(1).fillna(0.0)   # 次日生效, 无前视
        gross_ret = (pos * R.fillna(0.0)).sum(axis=1)
        hold = pos.ne(0) | pos.shift(1).fillna(0.0).ne(0)
        turnover = (pos - pos.shift(1).fillna(0.0)).abs().sum(axis=1)

        for cb in COST_BPS_SET:
            net = gross_ret - turnover * (cb / 1e4)
            nav = (1 + net).cumprod()
            p = perf(nav, net, label, cb)
            if p:
                p["nav"] = nav
                results[(label, cb)] = p

    # ---------- 表 1: 全期绩效 ----------
    print("\n" + "=" * 100)
    print("表 1  全期绩效 (成本敏感性)")
    print("=" * 100)
    print(f"{'策略':<14}{'成本bp':>8}{'年化收益':>11}{'年化波动':>11}"
          f"{'夏普':>8}{'最大回撤':>10}{'累计':>10}")
    print("-" * 100)
    for (label, cb), p in results.items():
        print(f"{pad_cn(label, 14)}{cb:>8.0f}{p['cagr']*100:>10.1f}%"
              f"{p['vol']*100:>10.1f}%{p['sharpe']:>8.2f}"
              f"{p['mdd']*100:>9.1f}%{p['total']*100:>9.1f}%")

    # ---------- 表 2: 24 个交易日(≈5 周)滚动窗口的收益分布 ----------
    print("\n" + "=" * 100)
    print(f"表 2  考核期直接对照: {WINDOW} 个交易日({WINDOW/5:.1f} 周)滚动窗口的净值涨跌分布")
    print("      这是「我这次可能考成什么样」的答案, 比看全期年化更有意义")
    print("=" * 100)
    print(f"{'策略':<14}{'成本':>5}{'P(>0)':>8}{'5%':>8}{'25%':>8}{'中位':>8}"
          f"{'75%':>8}{'95%':>8}{'最差':>8}{'最好':>8}")
    print("-" * 100)
    for cb in COST_BPS_SET:
        for label in ["TSMOM", "TSMOM-REV", "CARRY", "MIX70/30", "TSMOM+CARRY", "EQUAL-LONG"]:
            p = results.get((label, cb))
            if not p:
                continue
            nav = p["nav"]
            win = (nav.shift(-WINDOW) / nav - 1).dropna()
            if len(win) < N_LONG_WIN:
                continue
            q = win.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            print(f"{pad_cn(label, 14)}{cb:>5.0f}{(win > 0).mean()*100:>7.0f}%"
                  f"{q[0.05]*100:>7.1f}%{q[0.25]*100:>7.1f}%{q[0.5]*100:>7.1f}%"
                  f"{q[0.75]*100:>7.1f}%{q[0.95]*100:>7.1f}%"
                  f"{win.min()*100:>7.1f}%{win.max()*100:>7.1f}%")
        print()

    # ---------- 表 3: 逐年 ----------
    print("=" * 100)
    print("表 3  逐自然年收益 (6bp 成本) —— 看稳定性, 不要只看全期")
    print("=" * 100)
    for label in ["TSMOM", "TSMOM-REV", "CARRY", "MIX70/30", "TSMOM+CARRY", "EQUAL-LONG"]:
        p = results.get((label, 6.0))
        if not p:
            continue
        nav = p["nav"]
        yr = nav.groupby(nav.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        line = "  ".join(f"{y}:{v*100:+6.1f}%" for y, v in yr.items())
        print(f"{pad_cn(label, 14)}{line}")

    print("\n说明: 这是【组合层面】的研究回测, 成交按收盘价 + 换手成本近似;")
    print("      进入实盘模拟前, 还需用每日 runner 在 TqKq 上跑一遍确认可执行性。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
