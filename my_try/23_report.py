# -*- coding: utf-8 -*-
"""
绩效报告 / 每日快照 —— 考核期结束后用于写报告, 过程中用于监控。

三种用法:

  1. 每日快照(TqKq 排练账户, 收盘后跑):
        python my_try/23_report.py --snapshot

  2. 每日快照(考核平台, 自己抄权益数字进去, 不连交易接口):
        python my_try/23_report.py --manual 5123456 --note "第1周收盘"

  3. 出报告(读 snapshots.csv + 指令单, 生成图 + 绩效表):
        python my_try/23_report.py --report

产出:
  output/snapshots.csv   每行一天的账户快照
  output/report.png      净值曲线 + 回撤曲线
  output/report.txt      绩效汇总 + 逐周明细(同时在屏幕上打印)

为什么快照要单独存:
    考核平台的净值只能靠每天记一次才有历史。事后想补是补不出来的——
    这是整个流程里唯一无法重做的事, 所以放在每天收盘后第一件事。
"""
import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
SNAP = OUT / "snapshots.csv"
REPORT_TXT = OUT / "report.txt"
REPORT_PNG = OUT / "report.png"

SNAP_FIELDS = ["date", "ts", "equity", "balance", "available", "margin",
               "float_profit", "risk_ratio", "n_pos", "note"]


def read_snaps():
    if not SNAP.exists():
        return pd.DataFrame(columns=SNAP_FIELDS)
    df = pd.read_csv(SNAP, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def write_snap(row):
    new = not SNAP.exists()
    with SNAP.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SNAP_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def cmd_snapshot(note=""):
    """连 TqKq 排练账户, 记录今天的权益/持仓/保证金。"""
    from tqsdk import TqApi, TqAuth, TqKq
    from tq_auth import TQ_USER, TQ_PASS

    api = TqApi(TqKq(), auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        acc = api.get_account()
        pos = api.get_position()
        margin = float(sum(p.margin for p in pos.values() if p.margin == p.margin))
        fprofit = float(sum(p.float_profit for p in pos.values()
                            if p.float_profit == p.float_profit))
        npos = sum(1 for p in pos.values()
                   if (p.pos_long or 0) + (p.pos_short or 0) > 0)
        row = dict(date=date.today().isoformat(),
                   ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   equity=round(float(acc.balance), 2),
                   balance=round(float(acc.balance), 2),
                   available=round(float(acc.available), 2),
                   margin=round(margin, 2),
                   float_profit=round(fprofit, 2),
                   risk_ratio=round(float(acc.risk_ratio or 0), 4),
                   n_pos=npos, note=note)
        write_snap(row)
        print(f"已记录 {row['date']}: 权益 {row['equity']:,.0f}, "
              f"持仓 {npos} 个, 保证金 {margin:,.0f}, 浮动盈亏 {fprofit:,.0f}")
        return 0
    finally:
        api.close()


def cmd_manual(equity, note=""):
    """不连交易接口, 手工记录考核平台的权益。"""
    prev = read_snaps()
    equity = float(equity)
    row = dict(date=date.today().isoformat(),
               ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               equity=round(equity, 2), balance="", available="", margin="",
               float_profit="", risk_ratio="", n_pos="", note=note)
    write_snap(row)
    print(f"已手工记录 {row['date']}: 权益 {equity:,.0f}"
          + (f" (上一记录 {prev['equity'].iloc[-1]:,.0f}, "
             f"变动 {equity/float(prev['equity'].iloc[-1])-1:+.2%})" if len(prev) else ""))
    return 0


def cmd_report():
    df = read_snaps()
    if len(df) < 2:
        print(f"快照只有 {len(df)} 条, 至少要 2 天的数据才能出报告。")
        print("先每天收盘后运行: python my_try/23_report.py --snapshot")
        return 1

    eq = df.set_index("date")["equity"].astype(float)
    ret = eq.pct_change().dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    cagr = (1 + total) ** (365 / days) - 1
    vol = ret.std(ddof=1) * (252 ** 0.5) if len(ret) > 1 else float("nan")
    sharpe = (ret.mean() * 252) / vol if vol and vol > 0 else float("nan")
    dd = (eq / eq.cummax() - 1)
    mdd = dd.min()

    # 逐周(周五)明细: 看每周调仓后的效果
    weekly = eq.resample("W-FRI").last().dropna()
    wk_ret = weekly.pct_change().dropna()

    lines = []
    A = lines.append
    A("=" * 78)
    A(f"考核期绩效报告   数据区间 {eq.index[0].date()} ~ {eq.index[-1].date()} ({days} 天)")
    A("=" * 78)
    A(f"期初权益      {eq.iloc[0]:>14,.0f}")
    A(f"期末权益      {eq.iloc[-1]:>14,.0f}")
    A(f"区间收益      {total:>13.2%}")
    A(f"年化收益      {cagr:>13.2%}")
    A(f"年化波动      {vol:>13.2%}")
    A(f"夏普(2%无风险){sharpe:>13.2f}")
    A(f"最大回撤      {mdd:>13.2%}")
    A(f"胜率(按周)    {(wk_ret > 0).mean():>13.1%}   ({len(wk_ret)} 周)")
    A("")
    A("逐周明细(每周五收盘):")
    A(f"{'日期':<14}{'权益':>14}{'周收益':>10}{'累计回撤':>11}  备注")
    A("-" * 78)
    for i, (d, v) in enumerate(weekly.items()):
        r = (v / weekly.iloc[i - 1] - 1) if i > 0 else 0.0
        note = ""
        m = df[df["date"] == d]
        if len(m) and isinstance(m.iloc[-1]["note"], str):
            note = m.iloc[-1]["note"]
        A(f"{str(d.date()):<14}{v:>14,.0f}{r:>9.2%}"
          f"{(v / eq.cummax().loc[:d].max() - 1):>11.2%}  {note}")

    orders = sorted(OUT.glob("orders_*.csv"))
    if orders:
        last = orders[-1]
        try:
            od = pd.read_csv(last, encoding="utf-8-sig")
            A("")
            A(f"最近一张指令单: {last.name}  "
              f"{od['品种'].nunique()} 个品种 / {len(od)} 条腿")
        except Exception:
            pass

    text = "\n".join(lines)
    REPORT_TXT.write_text(text, encoding="utf-8")
    print(text)

    # ---------- 画图 ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                       gridspec_kw={"height_ratios": [3, 1]})
        ax1.plot(eq.index, eq.values, lw=2, color="#1f4e79")
        ax1.axhline(eq.iloc[0], ls="--", lw=1, color="gray")
        ax1.set_title(f"考核期净值  {total:+.2%}  最大回撤 {mdd:.2%}")
        ax1.set_ylabel("权益(元)")
        ax1.grid(alpha=0.3)
        ax2.fill_between(dd.index, dd.values * 100, 0, color="#c00000", alpha=0.4)
        ax2.set_ylabel("回撤(%)")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORT_PNG, dpi=130)
        print(f"\n图已保存: {REPORT_PNG}")
    except Exception as e:
        print(f"\n画图失败(不影响文字报告): {e}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", action="store_true", help="连 TqKq 记录今天快照")
    ap.add_argument("--manual", type=float, default=None, help="手工记录今天权益(元)")
    ap.add_argument("--note", default="", help="备注, 如 '第1周收盘'")
    ap.add_argument("--report", action="store_true", help="生成报告")
    args = ap.parse_args()

    if args.snapshot:
        return cmd_snapshot(args.note)
    if args.manual is not None:
        return cmd_manual(args.manual, args.note)
    if args.report:
        return cmd_report()
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
