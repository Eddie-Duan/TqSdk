# -*- coding: utf-8 -*-
"""
期限结构 / 基差检验 —— 换一个"有理论锚"的信号源。

运行:  python my_try/11_basis_test.py      (只读数据, 不做回测)

为什么做这个(前因):
    08~10 用四份证据证明了"纯价格派生的趋势策略"很弱, 而且无法实时利用。
    09 也指出: 要改变结果, 必须换信息源。
    本脚本转向第 3 层"理论锚"—— 期限结构。

理论依据(持有成本模型):
    F = S · exp( (r + u - y) · T )
      r 资金成本, u 仓储成本, y 便利收益(convenience yield)

    近月 > 远月  ->  Backwardation: 当前现货紧张, 便利收益高
    近月 < 远月  ->  Contango:     当前供给充裕, 库存高

    关键推论: **价差应该比价格本身更"均值回归"**。
    因为价格可以一路涨(趋势), 但价差偏离过大时, 会通过
    库存搬运(实物交割/仓储套利)被强行拉回 —— 这是有实物约束的。

所以要检验的假设:
    假设: 价差的 VR < 1 (均值回归), 与价格 VR > 1 (弱趋势) 形成对比。
    检验: ① 价差自身的方差比  ② 价差极值后的未来变化(严格无前视)
          ③ 潜在利润能不能覆盖交易成本

两个数据源(互为验证):
    A. 真实合约价差: 近月合约 - 次近月合约   (干净, 能直接交易, 但历史只有几个月)
    B. 主连 - 指数:  KQ.m@ - KQ.i@           (可回溯 4.75 年, 但是个代理指标)

注意: 成交仍是简化假设(收盘价、固定 1 手、每手固定手续费)。
"""
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区 ====================
SYM_NEAR = "SHFE.rb2610"        # 近月真实合约
SYM_FAR = "SHFE.rb2701"         # 次近月真实合约
SYM_MAIN = "KQ.m@SHFE.rb"       # 主连
SYM_INDEX = "KQ.i@SHFE.rb"      # 指数
ALL_SYMS = [SYM_NEAR, SYM_FAR, SYM_MAIN, SYM_INDEX]

DURATION = 60 * 60
DATA_LEN = 8000
VOLUME_MULTIPLE = 10            # 螺纹 10 吨/手
FEE_PER_LOT = 3.0               # 元/手
FWD_H = 40                      # 未来 H 小时
PCT_WIN_MAX = 500               # 滚动分位窗口上限(短序列会自动缩小)
VR_QS = [2, 5, 10, 20, 40, 80, 160]


def vr_stat(x, q):
    """Lo-MacKinlay 方差比。x 可以是序列本身(价差)或对数价格 —— 内部做一阶差分。
       VR < 1 -> 均值回归;  VR > 1 -> 趋势。"""
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
    """out[i] = 当前值在过去 w 个样本中的分位(0~1)。只用历史, 无前视偏差。"""
    out = np.full(len(v), np.nan)
    for i in range(w, len(v)):
        win = v[i - w + 1:i + 1]
        out[i] = float((win <= v[i]).mean())
    return out


def wait_all(api, frames, timeout=180):
    """等所有序列都拿到数据(数量稳定)。"""
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


def merge_pair(frames, sym_a, sym_b):
    """把两个合约的收盘价按时间对齐(取交集)。"""
    a = frames[sym_a][["datetime", "close"]].dropna().rename(columns={"close": "a"})
    b = frames[sym_b][["datetime", "close"]].dropna().rename(columns={"close": "b"})
    m = pd.merge(a, b, on="datetime", how="inner").sort_values("datetime").reset_index(drop=True)
    return m


def analyze(title, note, m, unit_label):
    """对一个价差序列做完整检验。"""
    if len(m) < 300:
        print(f"\n\n{'=' * 96}\n{title}\n{'=' * 96}\n数据不足({len(m)} 根), 跳过。")
        return

    dt = pd.to_datetime(m["datetime"], unit="ns") + pd.Timedelta(hours=8)
    level = (m["a"] - m["b"]).to_numpy(dtype=float)
    years = (dt.iloc[-1] - dt.iloc[0]).days / 365.25

    print()
    print("=" * 96)
    print(f"{title}")
    print(f"  {note}")
    print(f"  区间 {dt.iloc[0]:%Y-%m-%d} ~ {dt.iloc[-1]:%Y-%m-%d}   "
          f"{len(level)} 根K线 / {years:.2f} 年")
    print("=" * 96)

    # ---------- ① 描述统计 ----------
    lo_p, hi_p = np.percentile(level, [10, 90])
    print(f"价差(近月-远月)  均值 {level.mean():+8.2f}{unit_label}   "
          f"标准差 {level.std():.2f}   最低 {level.min():+.2f}   最高 {level.max():+.2f}")
    print(f"  当前 {level[-1]:+.2f}{unit_label}   过去分位 "
          f"{(level <= level[-1]).mean() * 100:.0f}%")
    print(f"  Contango    (近月<远月): {(level < 0).mean() * 100:5.1f}%")
    print(f"  Backwardation(近月>远月): {(level > 0).mean() * 100:5.1f}%")

    # ---------- ② 价差自身的方差比 ----------
    print()
    print(f"  价差的方差比检验  (VR < 1 = 均值回归, 与价格 VR > 1 形成对比)")
    print(f"  {'q(小时)':<10}{'VR(q)':>9}{'z值':>9}{'显著':>6}   方向")
    print("  " + "-" * 62)
    for q in VR_QS:
        if len(level) <= q + 2:
            continue
        vr, z = vr_stat(level, q)
        sig = "  *" if abs(z) > 1.96 else ""
        bar = ("+" if vr >= 1 else "-") * min(int(round(abs(vr - 1) * 40)), 30)
        print(f"  {q:<10}{vr:>9.3f}{z:>+9.2f}{sig:>6}   {bar}")

    # ---------- ③ 极值之后会不会回归(严格无前视) ----------
    w = min(PCT_WIN_MAX, max(30, len(level) // 4))
    rollp = rolling_pctile(level, w)
    fwd = np.full(len(level), np.nan)
    if len(level) > FWD_H + 1:
        fwd[:-FWD_H] = level[FWD_H:] - level[:-FWD_H]

    ok = ~np.isnan(rollp) & ~np.isnan(fwd)
    p, f = rollp[ok], fwd[ok]
    print()
    print(f"  滚动分位 → 未来 {FWD_H} 小时价差变化   (窗口 {w} 根, 只用历史)")
    print(f"  {'当前分位':<14}{'样本数':>8}{'未来变化均值':>14}{'反向占比':>10}")
    print("  " + "-" * 62)
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.001]
    labels = ["0-10% 极低", "10-25%", "25-50%", "50-75%", "75-90%", "90-100% 极高"]
    grp = pd.cut(pd.Series(p), bins, labels=labels, include_lowest=True)
    tb = pd.DataFrame({"g": grp.to_numpy(), "f": f}).groupby("g", observed=True)
    for gi, part in tb:
        # 均值回归的预期: 分位低 -> 未来上涨(>0); 分位高 -> 未来下跌(<0)
        rev = (part["f"] > 0).mean() if gi in ("0-10% 极低", "10-25%") else (part["f"] < 0).mean()
        print(f"  {str(gi):<14}{len(part):>8}{part['f'].mean():>+13.2f}{unit_label}"
              f"{rev * 100:>9.1f}%")

    if len(p) > 10:
        r, pval = pearsonr(p, f)
        print()
        print(f"  corr(当前分位, 未来{FWD_H}h价差变化) = {r:+.4f}   p = {pval:.4f}"
              f"   ({'显著' if pval < 0.05 else '不显著'})")
        print(f"  均值回归预期: r < 0。|r| 越大、越显著, 说明偏离后回归越明确。")

    # ---------- ④ 潜在利润 vs 成本 ----------
    profit = (hi_p - lo_p) * VOLUME_MULTIPLE          # 从10%分位到90%分位的价差幅度, 元/手
    cost = 4 * FEE_PER_LOT                            # 开+平 × 两个合约
    print()
    print(f"  成本检查: 10%~90% 分位的价差幅度 = {profit:,.0f} 元/手; "
          f"一次往返成本 ≈ {cost:,.0f} 元/手  ->  "
          f"{'利润空间充足' if profit > 5 * cost else '利润空间勉强' if profit > 2 * cost else '利润空间不足'}")


def main():
    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        frames = {s: api.get_kline_serial(s, DURATION, data_length=DATA_LEN) for s in ALL_SYMS}
        print(f"正在下载 {len(ALL_SYMS)} 个序列的 {DATA_LEN} 根 {DURATION // 60} 分钟K线 ...")
        wait_all(api, list(frames.values()))
        merged_contract = merge_pair(frames, SYM_NEAR, SYM_FAR)
        merged_proxy = merge_pair(frames, SYM_MAIN, SYM_INDEX)
    finally:
        api.close()

    print()
    print("=" * 96)
    print("期限结构 / 基差检验   标的: 螺纹钢 (SHFE.rb)")
    print("假设: 价差比价格本身更均值回归(有实物交割的约束), 因此更适合做策略")
    print("=" * 96)

    analyze(f"A. 真实合约价差: {SYM_NEAR} - {SYM_FAR}",
            "两合约都是可交易的真实合约, 数据干净, 但重叠历史较短",
            merged_contract, "元/吨")

    analyze(f"B. 主连 - 指数 代理: {SYM_MAIN} - {SYM_INDEX}",
            "可回溯 4.75 年; 指数的具体加权方式由服务器决定, 所以它只是"
            "「主力相对全月份均价」的代理指标",
            merged_proxy, "元/吨")

    print()
    print("=" * 96)
    print("判读要点")
    print("=" * 96)
    print("  1) 若价差的 VR < 1 且显著, 而 09 里价格的 VR > 1  ->  "
          "印证「价格有趋势、价差会回归」, 这是期限结构策略的理论基础。")
    print("  2) 看「90-100% 极高」那一组的未来变化是否显著为负 —— "
          "这是可以真正交易的信号。")
    print("  3) 最后看成本: 价差幅度必须显著大于往返手续费, 否则回归赚的钱不够付成本。")
    print("  4) 诚实提醒: A 序列只有几个月, 统计效力弱; B 序列长但是代理指标, "
          "两者结论一致才可信。")


if __name__ == "__main__":
    main()
