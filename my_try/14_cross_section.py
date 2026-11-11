# -*- coding: utf-8 -*-
"""
横截面检验 —— 把样本从"14 段"换成"14 期 x N 品种"。

运行:  python my_try/14_cross_section.py      (只读数据, 不做回测)

为什么做这个(前因):
    08~13 全部是【时间序列】检验: 一个品种 vs 它自己的历史。
    这带来一个绕不过去的统计瓶颈: 小时线 8000 根上限 -> 4.75 年 -> 段级有效
    样本只有 14 段。13 号算出的"成立段 7/14"根本无法判读, 因为段数太少。

    但商品期货有一个天然的横截面: 同一时点上有 60 个品种。
    样本于是从 (4.75 年 x 1 品种) 变成 (几十期 x N 品种), 量级完全不同,
    而且可以绕开"挑品种"这种过拟合。

    更重要的是, 横截面问的是 13 号问不到的问题:
        "期限结构(展期收益)能不能预测【品种之间】的收益差",
    而不是"某一个品种能不能预测它自己"。
    文献上这叫 carry / roll yield 因子, 是商品领域少数有理论锚的因子之一。

做法:
  1. 品种池: 换月表里所有商品期货(剔除金融), 按主力成交量取前 UNIVERSE_N 个
  2. 每月第一个交易日调仓:
        carry(i,t) = (P近(i,t) / P远(i,t) - 1) / (合约月差/12)     年化展期收益
     P近 = 当期主力合约, P远 = 下一期主力合约(即当期次主力), 与 12/13 号同口径
  3. 收益: 可交易连续(ratio-adjusted)日线收益 —— 换月日用新合约自己的日收益,
     段内用同一合约的日收益, 复利累积。这才是"持有主力并展期"的真实 P&L。
     (第一版误用 KQ.m@ 主连收益, 见下方【重要修正】)
  4. 检验: 每期横截面 IC (Spearman) + N 分组单调性 + 多空组合净值
     同时跑一个 momentum 因子(过去 12 月收益, 跳过最近 1 月)做对照
  5. 成本: turnover x (滑点tick x price_tick / price), 做 0/1/2 tick 敏感性

【重要修正 · 主连口径 vs 可交易口径】
    第一版用 KQ.m@ 主连日线算收益, 得到 carry 的 IC = -0.139 (t = -3.50),
    即"高 carry 品种下期跑输", 与理论符号完全相反。这是【假象】:
      主连在换月日直接跳到新合约的绝对价格, 于是每滚一次就凭空多出
      (P远/P近 - 1) 这一段; 而真实滚动是等名义价值换仓、损益中性。
      可推出:  主连收益 = 真实收益 - carry
    用一条"已经减掉 carry"的序列去回归 carry, 必然得到负 IC。
    本脚本同时输出两个口径, 请只看【可交易口径】那一行。

已知让步(必须明说):
  * 品种池按【当前】成交量选, 有轻微前视 / 存活偏差。缓解: 同时要求 2022 年以来
    至少有 MIN_SEG 个完整换月段。
  * 成本仍是假设值(1 tick 滑点), 不是实测数据 —— 这是 11~14 号共同的天花板。
  * 远腿当日无成交时收盘价是"陈旧价格"(stale price), 会污染 carry 与收益的关系。
    脚本统计远腿零成交占比, 并给出去掉零成交品种后的 IC 做对照。
"""
import gc
import pickle
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
CONT_TABLE_URL = "https://files.shinnytech.com/continuous_table.json"
EX_OK = {"SHFE", "DCE", "CZCE", "INE", "GFEX"}

START_DATE = date(2022, 1, 1)     # 调仓序列起点
UNIVERSE_N = 30                   # 取成交量前 N 个品种
MIN_VOL = 20000                   # 主力成交量门槛(手)
MIN_SEG = 6                       # 2022 年以来至少要有几个完整换月段

DAY = 24 * 60 * 60
DATA_LEN = 2000                   # 日线 2000 根(合约寿命只有约 250 根, 足够)
BATCH = 100                       # 每批订阅多少个序列
CACHE = Path(__file__).parent / "output" / "cache_14_daily.pkl"

N_GROUPS = 5                      # 分 5 组看单调性
TOP_FRAC = 0.30                   # 多空各取前/后 30%
SLIP_SET = [0, 1, 2]              # 成本敏感性: 每次换仓每条腿的滑点(tick)


def pad_cn(text, width):
    """按终端显示宽度补齐(中文按 2 计)。"""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    return text + " " * max(0, width - w)


# ==================== 合约代码 -> 年月 ====================
def contract_month(sym):
    """把合约代码变成 年*12+月 的整数。支持 DCE/SHFE 四位(A2701)与 CZCE 三位(TA701)。"""
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
    if not (1 <= m <= 12):
        return None
    return y * 12 + m


def get_rollovers(table, key):
    rows = table.get(key)
    if not rows:
        return None
    out = [(datetime.strptime(d, "%Y%m%d").date(), sym) for d, sym in rows]
    out.sort(key=lambda x: x[0])
    return out


def wait_all(api, frames, timeout=900):
    """等所有序列数据稳定。"""
    last, stable, end = -1, 0, time.time() + timeout
    while time.time() < end:
        api.wait_update(deadline=time.time() + 1)
        cnt = sum(int(f["close"].notna().sum()) for f in frames)
        if cnt == last and cnt > 100:
            stable += 1
            if stable >= 4:
                return
        else:
            stable = 0
        last = cnt


# ==================== 主流程 ====================
def main():
    print("=" * 112)
    print("横截面检验 · 期限结构(carry) vs 动量(momentum)         (只读数据, 不做回测)")
    print("=" * 112)

    table = requests.get(CONT_TABLE_URL, timeout=30).json()
    keys = [k for k in table if k.split(".")[0] in EX_OK]
    print(f"换月表总键数 {len(table)}, 商品期货键数 {len(keys)}")

    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        # ---------- 1. 选品种池 ----------
        quotes = {k: api.get_quote("KQ.m@" + k) for k in keys}
        for _ in range(300):
            api.wait_update()
            if all(q.datetime for q in quotes.values()):
                break

        cand = []
        for k in keys:
            q = quotes[k]
            roll = get_rollovers(table, k)
            if not roll:
                continue
            nseg = sum(1 for i in range(len(roll) - 1)
                       if roll[i + 1][0] > START_DATE and roll[i][0] < date.today())
            vol = float(q.volume) if q.volume else 0.0
            tick = float(q.price_tick) if q.price_tick else 0.0
            if vol < MIN_VOL or nseg < MIN_SEG or tick <= 0:
                continue
            cand.append((k, vol, nseg, tick))
        cand.sort(key=lambda x: -x[1])
        universe = cand[:UNIVERSE_N]
        tick_of = {k: t for k, _, _, t in universe}

        print()
        print(f"选入品种池 {len(universe)} 个 (成交量>={MIN_VOL}, 换月段>={MIN_SEG}):")
        line = "  "
        for k, vol, nseg, tick in universe:
            line += f"{k}({vol/10000:.0f}万手/{nseg}段)  "
            if len(line) > 100:
                print(line)
                line = "  "
        if len(line.strip()):
            print(line)

        # ---------- 2. 收集要下载的合约 ----------
        need = set()
        for k, _, _, _ in universe:
            roll = get_rollovers(table, k)
            for i in range(len(roll) - 1):
                # 只看 2022 至今真实发生过的段; roll[i] 必须先于今天(表里含未来换月点)
                if roll[i + 1][0] <= START_DATE or roll[i][0] >= date.today():
                    continue
                for sym in (roll[i][1], roll[i + 1][1]):
                    if contract_month(sym):
                        need.add(sym)
        main_syms = ["KQ.m@" + k for k, _, _, _ in universe]
        all_syms = sorted(need) + main_syms
        print(f"\n需要下载 {len(all_syms)} 个日线序列 ({len(need)} 个真实合约 + "
              f"{len(main_syms)} 个主连), 分批 {BATCH} 个 ...")

        # ---------- 3. 分批下载(命中缓存则跳过, 重跑调参时省 10 分钟) ----------
        close, volu = {}, {}
        if CACHE.exists():
            try:
                blob = pickle.loads(CACHE.read_bytes())
                if set(all_syms) <= set(blob["syms"]):
                    close, volu = blob["close"], blob["volu"]
                    print(f"命中缓存 {CACHE.name}, 跳过下载 ({len(close)} 个序列)")
            except Exception as e:
                print(f"缓存不可用({type(e).__name__}), 重新下载")

        t0 = time.time()
        for b in ([] if close else range(0, len(all_syms), BATCH)):
            batch = all_syms[b:b + BATCH]
            try:
                frames = {s: api.get_kline_serial(s, DAY, data_length=DATA_LEN) for s in batch}
                wait_all(api, list(frames.values()))
                for s in batch:
                    f = frames[s]
                    df = f[["datetime", "close", "volume"]].dropna(subset=["close"])
                    if len(df) == 0:
                        continue
                    dt = (pd.to_datetime(df["datetime"], unit="ns") + pd.Timedelta(hours=8)).dt.normalize()
                    close[s] = pd.Series(df["close"].to_numpy(float), index=dt.to_numpy())
                    volu[s] = pd.Series(df["volume"].to_numpy(float), index=dt.to_numpy())
                    close[s] = close[s][~close[s].index.duplicated(keep="last")]
                    volu[s] = volu[s][~volu[s].index.duplicated(keep="last")]
                del frames
                gc.collect()
            except Exception as e:
                print(f"  批 {b//BATCH+1} 失败: {type(e).__name__}: {e}")
            done = min(b + BATCH, len(all_syms))
            print(f"  进度 {done}/{len(all_syms)}  已用 {time.time()-t0:.0f}s")

        if close:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_bytes(pickle.dumps({"syms": all_syms, "close": close, "volu": volu}))
            print(f"已写入缓存 {CACHE}")
    finally:
        api.close()

    got = sum(1 for s in need if s in close)
    print(f"\n下载完成: 真实合约 {got}/{len(need)}, 主连 {sum(1 for s in main_syms if s in close)}/{len(main_syms)}")

    # ---------- 4. 构造调仓月度网格 ----------
    cal = set()
    for s in main_syms:
        if s in close:
            cal.update(close[s].index)
    cal = pd.DatetimeIndex(sorted(cal))
    cal = cal[cal >= pd.Timestamp(START_DATE)]
    if len(cal) < 30:
        print("交易日太少, 退出")
        return

    # 每月第一个交易日
    rebal = []
    seen = set()
    for d in cal:
        ym = (d.year, d.month)
        if ym not in seen:
            seen.add(ym)
            rebal.append(d)
    rebal = pd.DatetimeIndex(rebal)
    print(f"调仓网格: {len(rebal)} 期, {rebal[0].date()} ~ {rebal[-1].date()}")

    # ---------- 5. 两条收益口径 ----------
    # 口径 A: KQ.m@ 主连(有换月跳空) —— 【错误】口径, 保留只为对照展示假象
    # 口径 B: 可交易连续(ratio-adjusted) —— 换月日用【新合约自己】的日收益,
    #   段内用同一合约的日收益, 复利累积。这才是持有主力并展期的真实 P&L。
    roll_map = {k: get_rollovers(table, k) for k, _, _, _ in universe}

    px_kq = pd.DataFrame({k: close.get("KQ.m@" + k, pd.Series(dtype=float)).reindex(rebal)
                          for k, _, _, _ in universe})

    idx_tr = pd.DataFrame(index=rebal, columns=px_kq.columns, dtype=float)
    for k in px_kq.columns:
        roll = roll_map[k]
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
        dts, rets = [], []
        for j in range(1, len(days)):
            d, dp = days[j], days[j - 1]
            sn = close.get(main_by_date[d])
            if sn is None or d not in sn.index or dp not in sn.index:
                continue
            v0, v1 = float(sn.loc[dp]), float(sn.loc[d])
            if v0 <= 0:
                continue
            dts.append(d)
            rets.append(v1 / v0 - 1.0)
        if len(rets) < 20:
            continue
        lev = (1.0 + pd.Series(rets, index=pd.DatetimeIndex(dts))).cumprod()
        idx_tr[k] = lev.reindex(rebal, method="ffill")

    fwd = idx_tr.shift(-1) / idx_tr - 1.0              # 【真实】下一期收益
    fwd_kq = px_kq.shift(-1) / px_kq - 1.0             # 【假象】主连口径收益
    mom = idx_tr.shift(1) / idx_tr.shift(12) - 1.0     # 过去 12 月动量, 跳过最近 1 月
    px = pd.DataFrame(index=rebal, columns=px_kq.columns, dtype=float)   # 当期主力收盘价

    # ---------- 6. 每期算 carry ----------
    near_at = {}          # (调仓日, 品种) -> 当期主力合约, 用于算展期成本
    carry = pd.DataFrame(index=rebal, columns=px.columns, dtype=float)
    far_vol0 = pd.DataFrame(index=rebal, columns=px.columns, dtype=float)
    for k in px.columns:
        roll = roll_map[k]
        for t in rebal:
            td = t.date()
            idx = None
            for i in range(len(roll) - 1):
                if roll[i][0] <= td < roll[i + 1][0]:
                    idx = i
                    break
            if idx is None:
                continue
            near, far = roll[idx][1], roll[idx + 1][1]
            sn, sf = close.get(near), close.get(far)
            if sn is None or sf is None or t not in sn.index or t not in sf.index:
                continue
            pn, pf = float(sn.loc[t]), float(sf.loc[t])
            gap = contract_month(far) - contract_month(near)
            if pn <= 0 or pf <= 0 or not (1 <= gap <= 18):
                continue
            carry.loc[t, k] = (pn / pf - 1.0) / (gap / 12.0)
            far_vol0.loc[t, k] = (float(volu[far].loc[t])
                                  if (far in volu and t in volu[far].index) else np.nan)
            near_at[(t, k)] = near
            px.loc[t, k] = pn

    print(f"carry 覆盖: 平均每期 {carry.notna().sum(axis=1).mean():.1f} 个品种 "
          f"(最少 {carry.notna().sum(axis=1).min()}, 最多 {carry.notna().sum(axis=1).max()})")

    # ==================== 表 1: 因子 IC ====================
    def sp(a, b):
        """Spearman = 秩上的 Pearson。不依赖 scipy 版本。"""
        return float(a.rank().corr(b.rank()))

    def ic_series(fac, ret=None):
        r_all = fwd if ret is None else ret
        out = []
        for t in rebal:
            f = fac.loc[t]
            r = r_all.loc[t]
            ok = f.notna() & r.notna()
            if ok.sum() < 6:
                out.append(np.nan)
                continue
            out.append(sp(f[ok], r[ok]))
        return pd.Series(out, index=rebal)

    ic_carry = ic_series(carry)
    ic_carry_kq = ic_series(carry, fwd_kq)
    ic_mom = ic_series(mom)
    ok_set = carry.notna() & fwd.notna() & (far_vol0.fillna(0) > 0)
    ic_carry_liq = ic_series(carry.where(ok_set))

    def ic_stats(ic, name):
        v = ic.dropna()
        n = len(v)
        if n < 5:
            return (name, n, np.nan, np.nan, np.nan)
        m, s = float(v.mean()), float(v.std(ddof=1))
        return (name, n, m, m / (s / np.sqrt(n)) if s > 0 else np.nan, float((v > 0).mean() * 100))

    print()
    print("=" * 112)
    print("表 1  横截面 IC (Spearman: 因子值 vs 下一期收益)")
    print("  IC>0 表示因子值高的品种, 下期收益确实更高。t 值按 IC 序列算。")
    print("  去掉远腿零成交品种 = 剔除'陈旧价格'污染后的版本")
    print("=" * 112)
    print(f"{pad_cn('因子', 22)}{'期数':>6}{'IC 均值':>11}{'t 值':>8}{'IC>0 比例':>11}")
    print("-" * 112)
    for rec in (ic_stats(ic_carry, "carry (可交易口径)"),
                ic_stats(ic_carry_kq, "carry (主连口径, 假象)"),
                ic_stats(ic_mom, "momentum (12-1)"),
                ic_stats(ic_carry_liq, "carry (去零成交)")):
        name, n, m, t, pos = rec
        print(f"{pad_cn(name, 22)}{n:>6}{m:>+11.4f}{t:>8.2f}{pos:>10.1f}%")

    # ==================== 表 2: 分组单调性 ====================
    print()
    print("=" * 112)
    print("表 2  分组平均下期收益 (按因子排 N 组, 等权; 单位 %)")
    print("=" * 112)
    print(f"{pad_cn('因子', 22)}" + "".join(f"{'Q'+str(i+1):>10}" for i in range(N_GROUPS))
          + f"{'Q高-Q低':>10}{'>0比例':>10}")
    print("-" * 112)

    def group_table(fac, name, ret=None):
        r_all = fwd if ret is None else ret
        rows = []
        for t in rebal:
            f, r = fac.loc[t], r_all.loc[t]
            ok = f.notna() & r.notna()
            if ok.sum() < N_GROUPS * 2:
                continue
            f, r = f[ok], r[ok]
            try:
                g = pd.qcut(f.rank(method="first"), N_GROUPS, labels=False)
            except Exception:
                continue
            rows.append([r[g == i].mean() * 100 for i in range(N_GROUPS)])
        if not rows:
            print(f"{pad_cn(name, 22)}  样本不足")
            return
        arr = np.array(rows)
        mean = arr.mean(axis=0)
        hi_lo = arr[:, -1] - arr[:, 0]
        print(f"{pad_cn(name, 22)}" + "".join(f"{v:>+10.3f}" for v in mean)
              + f"{hi_lo.mean():>+10.3f}{(hi_lo > 0).mean()*100:>9.0f}%")

    group_table(carry, "carry (可交易口径)")
    group_table(carry, "carry (主连口径, 假象)", ret=fwd_kq)
    group_table(mom, "momentum (12-1)")
    group_table(carry.where(ok_set), "carry (去零成交)")

    # ==================== 表 3: 多空组合 ====================
    print()
    print("=" * 112)
    print(f"表 3  多空组合 (每期做多因子前 {TOP_FRAC:.0%} / 做空后 {TOP_FRAC:.0%}, 等权, 总敞口 1)")
    print("=" * 112)
    print(f"{pad_cn('方案', 30)}{'期数':>6}{'月均(%)':>10}{'年化(%)':>10}{'年化波动':>10}"
          f"{'夏普':>8}{'最大回撤':>10}{'胜率':>8}")
    print("-" * 112)

    def ls_portfolio(fac, slip, name, quiet=False, ret=None):
        """多空组合。成本 = 名义成交额 x (滑点 tick x price_tick / price)。"""
        r_all = fwd if ret is None else ret
        rets, cost = [], []
        w_prev, main_prev = None, None
        for t in rebal:
            f, r = fac.loc[t], r_all.loc[t]
            ok = f.notna() & r.notna()
            if ok.sum() < 6:
                rets.append(np.nan)
                cost.append(0.0)
                continue
            f, r = f[ok], r[ok]
            k = max(1, int(round(len(f) * TOP_FRAC)))
            order = f.sort_values().index
            w = pd.Series(0.0, index=r.index)
            w[list(order[-k:])] = 1.0 / (2 * k)     # 做多高 carry
            w[list(order[:k])] = -1.0 / (2 * k)     # 做空低 carry
            rets.append(float((w * r).sum()))

            c = 0.0
            cur_main = {}
            for s in w.index:
                ps = float(px.loc[t, s])
                tk = tick_of.get(s, 0.0)
                cur = near_at.get((t, s))
                cur_main[s] = cur
                if not (ps > 0) or tk <= 0:
                    continue
                unit = slip * tk / ps
                prev_w = w_prev.get(s, 0.0) if w_prev is not None else 0.0
                traded = abs(w[s] - prev_w)
                prev_sym = main_prev.get(s) if main_prev else None
                if prev_sym is not None and cur is not None and prev_sym != cur:
                    traded += abs(w[s])             # 主力换月: 整仓滚一次
                c += traded * unit
            w_prev, main_prev = w, cur_main
            cost.append(c)

        g = pd.Series(rets, index=rebal, dtype=float)
        net = (g - pd.Series(cost, index=rebal, dtype=float)).dropna()
        if len(net) < 5:
            if not quiet:
                print(f"{pad_cn(name, 30)}  样本不足")
            return None
        cum = (1 + net).cumprod()
        ann = cum.iloc[-1] ** (12 / len(net)) - 1
        vol = net.std(ddof=1) * np.sqrt(12)
        dd = (cum / cum.cummax() - 1).min()
        sr = net.mean() / net.std(ddof=1) * np.sqrt(12) if net.std(ddof=1) > 0 else float("nan")
        if not quiet:
            print(f"{pad_cn(name, 30)}{len(net):>6}{net.mean()*100:>+10.3f}{ann*100:>+10.2f}"
                  f"{vol*100:>10.2f}{sr:>8.2f}{dd*100:>+10.2f}{(net > 0).mean()*100:>7.0f}%")
        return net

    for slip in SLIP_SET:
        ls_portfolio(carry, slip, f"carry  滑点{slip}tick")
    ls_portfolio(carry, 1, "carry (主连口径, 假象) 1tick", ret=fwd_kq)
    for slip in SLIP_SET:
        ls_portfolio(mom, slip, f"momentum  滑点{slip}tick")

    # ==================== 表 4: 逐年 ====================
    print()
    print("=" * 112)
    print("表 4  逐年收益 (滑点 1 tick)")
    print("=" * 112)
    res_c = ls_portfolio(carry, 1, "_carry", quiet=True)
    res_m = ls_portfolio(mom, 1, "_mom", quiet=True)
    years = sorted({t.year for t in rebal})
    print(f"{pad_cn('因子', 22)}" + "".join(f"{y:>10}" for y in years))
    print("-" * 112)
    for nm, res in (("carry", res_c), ("momentum", res_m)):
        if res is None:
            continue
        net = res.dropna()
        row = f"{pad_cn(nm, 22)}"
        for y in years:
            v = net[[t for t in net.index if t.year == y]]
            row += f"{((1+v).prod()-1)*100:>+10.2f}" if len(v) else f"{'-':>10}"
        print(row)

    # ==================== 表 5: 品种层面诊断 ====================
    print()
    print("=" * 112)
    print("表 5  品种层面 (是不是少数品种贡献了全部收益)")
    print("=" * 112)
    print(f"{pad_cn('品种', 10)}{'平均carry':>12}{'月均收益%':>12}{'收益标准差%':>13}"
          f"{'corr(可交易)':>14}{'corr(主连)':>12}{'远腿零成交':>12}")
    print("-" * 112)
    rows = []
    for k in px.columns:
        c = carry[k].dropna()
        if len(c) < 12:
            continue
        r = fwd[k].reindex(c.index)
        rk = fwd_kq[k].reindex(c.index)
        ok = r.notna()
        c2, r2 = c[ok], r[ok]
        if len(c2) < 12:
            continue
        rho = sp(c2, r2) if c2.nunique() > 2 else np.nan
        rho_kq = sp(c2, rk[ok]) if c2.nunique() > 2 else np.nan
        z0 = float((far_vol0[k].reindex(c2.index).fillna(0) <= 0).mean() * 100)
        rows.append((k, float(c2.mean()), float(r2.mean() * 100), float(r2.std(ddof=1) * 100),
                     rho, rho_kq, z0))
    rows.sort(key=lambda x: -x[2])
    for k, ac, mr, sd, rho, rho_kq, z0 in rows:
        print(f"{pad_cn(k, 10)}{ac:>+12.4f}{mr:>+12.3f}{sd:>13.2f}{rho:>+14.2f}"
              f"{rho_kq:>+12.2f}{z0:>11.0f}%")
    print()
    print("  远腿零成交占比高 = 该品种的 carry 值可能是陈旧价格, 该行结论不可信。")
    print("  corr(可交易) = 单品种时间序列上 carry 与真实收益的相关(与 13 号同性质)。")
    print("  corr(主连)   = 同一计算换成主连口径 —— 这一列普遍为负, 正是换月跳空的假象。")


if __name__ == "__main__":
    main()
