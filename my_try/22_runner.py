# -*- coding: utf-8 -*-
"""
carry 组合 —— 指令单生成器 / 模拟盘执行器。

两种用法:

  A. 生成团队指令单(考核用: 5 人各 100 万, 按组合计排名)
     python my_try/22_runner.py --members 5 --per-account 1000000
     输出: 屏幕上每人一张表 + output/orders_YYYYMMDD.csv
     不连交易接口, 只算信号和手数, 由 5 个人分别敲进考核平台。

  B. 在 TqKq 上排练(自己的账户, 自动下单)
     python my_try/22_runner.py                          # 干跑
     python my_try/22_runner.py --live                   # 真下单

设计要点:
  1. **每周调一次**(周五 14:45 收盘前)。carry 是年化慢信号, 排名一周内基本不变;
     44 条腿分给 5 个人, 每人约 9 条, 手工 10 分钟能完成。日频只会多付成本。
  2. **不依赖换月表文件**。continuous_table.json 会过期(实测落后两个月),
     近月取主连 KQ.m@<品种> 的 underlying_symbol, 远月取挂牌合约里下一个月份。
  3. **负载均衡拆账**: 按名义金额贪心分配并兼顾多空自平衡, 每人敞口和风险接近,
     组层面恰好还原完整横截面。
  4. 全量日志 -> output/runner_log.csv, 指令单 -> output/orders_*.csv。

策略(carry 横截面, 与 20_portfolio_backtest.py 同口径):
    carry = (P近月 / P远月 - 1) / (合约月差/12)          年化展期收益
    score = 横截面 carry 排名 -> [-1, 1]
    w     = score / 该品种年化波动, 归一化到总风险预算 LEVERAGE, 单品种上限 PER_INST_CAP
    手数  = w x 总权益 / (价格 x 合约乘数)

调度(Windows 计划任务, 每周五 14:45):
    schtasks /Create /SC WEEKLY /D FRI /TN "TqSdkCarry" ^
      /TR "cmd /c cd /d C:\\Users\\DUANboyang\\Desktop\\TqSdk && python my_try\\22_runner.py --members 5 --per-account 1000000" ^
      /ST 14:45
"""
import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tq_auth import TQ_USER, TQ_PASS  # noqa: E402

from tqsdk import TqApi, TqAuth, TqKq, TargetPosTask  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
LOG_CSV = OUT / "runner_log.csv"

DAY = 24 * 60 * 60
KLINE_LEN = 90
VOL_WIN = 60
VOL_FLOOR = 0.08
LEVERAGE = 4.0
PER_INST_CAP = 0.10
MIN_VOLUME = 5000
FALLBACK_MARGIN = 0.12
MIN_W = 0.015          # 权重绝对值低于此值的腿不交易(排名中间, 对组合无贡献)

# 品种池: 国内商品期货流动性最好的一批(想加减直接改这里)
UNIVERSE = [
    "SHFE.rb", "SHFE.hc", "SHFE.cu", "SHFE.al", "SHFE.zn", "SHFE.ni",
    "SHFE.au", "SHFE.ag", "SHFE.ru", "SHFE.bu", "SHFE.sp", "SHFE.fu", "SHFE.ss",
    "DCE.i", "DCE.j", "DCE.jm", "DCE.m", "DCE.y", "DCE.p", "DCE.c", "DCE.cs",
    "DCE.a", "DCE.jd", "DCE.eg", "DCE.l", "DCE.v", "DCE.pp", "DCE.eb",
    "CZCE.CF", "CZCE.SR", "CZCE.TA", "CZCE.MA", "CZCE.FG", "CZCE.SA",
    "CZCE.OI", "CZCE.RM", "CZCE.AP", "CZCE.CJ", "CZCE.UR", "CZCE.SF", "CZCE.SM",
    "INE.sc", "GFEX.si", "GFEX.lc",
]

CONTRACT_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


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


def log_row(row):
    new = not LOG_CSV.exists()
    with LOG_CSV.open("a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def build_contract_map(api):
    out = {}
    for ex in ("SHFE", "DCE", "CZCE", "INE", "GFEX"):
        try:
            syms = api.query_quotes(ins_class="FUTURE", exchange_id=ex)
        except Exception as e:
            print(f"  查询 {ex} 合约表失败: {e}")
            continue
        exmap = {}
        for s in syms:
            if "." not in s:
                continue
            m = CONTRACT_RE.match(s.split(".")[-1])
            if m:
                exmap.setdefault(m.group(1), []).append(s)
        out[ex] = exmap
    return out


def snake_assignment(n_items, n_members):
    """蛇形分配: 1,2,..,N, N,..,2,1 循环。让每人拿到的高低排名数量接近。"""
    seq = []
    while len(seq) < n_items:
        seq += list(range(n_members)) + list(range(n_members - 1, -1, -1))
    return seq[:n_items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=0,
                    help="团队人数; >1 时生成每人一张指令单(不下单)")
    ap.add_argument("--per-account", type=float, default=1_000_000,
                    help="每人账户权益(默认 100 万)")
    ap.add_argument("--live", action="store_true", help="真的下单到 TqKq(仅 members=1)")
    ap.add_argument("--lev", type=float, default=LEVERAGE, help="组合总风险预算倍数")
    ap.add_argument("--capital", type=float, default=0,
                    help="单账户按这个权益算手数; 不填用账户实际权益")
    ap.add_argument("--top", type=int, default=0, help="只做成交额前 N 个品种")
    ap.add_argument("--cap", type=float, default=PER_INST_CAP, help="单品种仓位上限")
    args = ap.parse_args()

    now = datetime.now()
    print("=" * 104)
    title = (f"团队指令单 · {args.members} 人 x {args.per_account/1e4:,.0f} 万 "
             f"= {args.members*args.per_account/1e4:,.0f} 万" if args.members > 1
             else "carry 组合执行器" + ("【实盘下单】" if args.live else "【干跑】"))
    print(f"{title}   时间 {now:%Y-%m-%d %H:%M:%S}   杠杆 {args.lev}x")
    print("=" * 104)

    api = TqApi(TqKq(), auth=TqAuth(TQ_USER, TQ_PASS))
    try:
        # ---------- 1. 挂牌合约表 ----------
        cmap = build_contract_map(api)
        listed = {ex: {v: sorted(syms, key=lambda s: contract_month(s) or 0)
                       for v, syms in m.items()} for ex, m in cmap.items()}
        print("合约表: " + ", ".join(f"{ex}:{sum(len(v) for v in m.values())}"
                                     for ex, m in listed.items()))

        # ---------- 2. 主连 -> 当期主力 ----------
        mains = {}
        for k in UNIVERSE:
            ex, var = k.split(".")
            if var not in listed.get(ex, {}):
                continue
            mains[k] = api.get_quote("KQ.m@" + k)
        print(f"品种 {len(mains)} 个, 等主连报价 ...")
        end = time.time() + 120
        while time.time() < end:
            api.wait_update(deadline=time.time() + 1)
            if all(m.underlying_symbol for m in mains.values()):
                break

        # ---------- 3. 近月/远月 ----------
        plan = {}
        for k, mq in mains.items():
            ex, var = k.split(".")
            near = mq.underlying_symbol
            if not near:
                continue
            nm = contract_month(near)
            far = None
            for s in listed.get(ex, {}).get(var, []):
                cm = contract_month(s)
                if cm and cm > nm:
                    far = s
                    break
            if far:
                plan[k] = (near, far)
        print(f"可算 carry 的品种 {len(plan)} 个, 订阅行情 ...")

        klines, quotes = {}, {}
        for k, (near, far) in plan.items():
            klines[near] = api.get_kline_serial(near, DAY, data_length=KLINE_LEN)
            quotes[near] = api.get_quote(near)
            quotes[far] = api.get_quote(far)

        end = time.time() + 300
        while time.time() < end:
            api.wait_update(deadline=time.time() + 1)
            if all(int(f["close"].notna().sum()) >= VOL_WIN for f in klines.values()):
                break
        ready = sum(1 for f in klines.values() if int(f["close"].notna().sum()) >= VOL_WIN)
        print(f"日线就绪 {ready}/{len(klines)}")

        # ---------- 4. 每品种指标 ----------
        rows = []
        for k, (near, far) in plan.items():
            close = klines[near]["close"].dropna()
            if len(close) < VOL_WIN + 5:
                continue
            sigma = max(float(close.pct_change().dropna().tail(VOL_WIN).std(ddof=1))
                        * np.sqrt(252), VOL_FLOOR)
            qn, qf = quotes[near], quotes[far]
            pn, pf = float(qn.last_price or 0), float(qf.last_price or 0)
            mult = float(qn.volume_multiple or 0)
            gap = contract_month(far) - contract_month(near)
            vol = float(qn.volume or 0)
            if pn <= 0 or pf <= 0 or mult <= 0 or not gap or not (1 <= gap <= 18):
                continue
            if vol < MIN_VOLUME:
                continue
            rows.append(dict(variety=k, near=near, far=far, price=pn, mult=mult,
                             sigma=sigma, carry=(pn / pf - 1.0) / (gap / 12.0),
                             volume=vol,
                             margin=float(qn.margin or 0) or pn * mult * FALLBACK_MARGIN))

        if len(rows) < 8:
            print(f"有效品种只有 {len(rows)} 个, 不足以构建横截面, 退出")
            return 1
        if args.top and args.top < len(rows):
            rows.sort(key=lambda r: -r["volume"])
            rows = rows[:args.top]
            print(f"按成交量只保留前 {args.top} 个品种")

        # ---------- 5. 排名 -> 风险权重 ----------
        acct = api.get_account()
        bal, avail = float(acct.balance or 0), float(acct.available or 0)
        total = (args.members * args.per_account if args.members > 1
                 else (args.capital if args.capital > 0 else bal))

        carry = np.array([r["carry"] for r in rows])
        rank = np.argsort(np.argsort(carry)) / (len(rows) - 1) - 0.5
        w = (rank / 0.5) / np.array([r["sigma"] for r in rows])
        w = w / np.abs(w).sum() * args.lev
        w = np.clip(w, -args.cap, args.cap)

        # 剔除两类无法有效下注的腿:
        #   a) 权重接近 0 的"噪声腿"(排名中间), 对组合无贡献却要吃手续费
        #   b) 一手的名义金额就已经超过目标预算的品种(如 au 一手 87 万),
        #      强行按 1 手下会严重超出单品种风险上限
        keep, dropped = [], []
        for i, (r, wi) in enumerate(zip(rows, w)):
            min_lot = r["price"] * r["mult"]
            target_n = abs(wi) * total
            if abs(wi) < MIN_W:
                dropped.append((r["variety"], f"权重仅 {wi:+.3f}"))
            elif target_n < min_lot * 0.75:
                dropped.append((r["variety"], f"一手需 {min_lot/1e4:.0f} 万 > 预算 {target_n/1e4:.0f} 万"))
            else:
                keep.append(i)
        if dropped:
            print(f"\n剔除 {len(dropped)} 个无法有效下注的品种:")
            for v, why in dropped:
                print(f"  - {v:<12} {why}")
        rows = [rows[i] for i in keep]
        w = w[keep]
        w = w / np.abs(w).sum() * args.lev
        w = np.clip(w, -args.cap, args.cap)
        carry = np.array([r["carry"] for r in rows])

        # ---------- 6. 团队模式: 每人一张指令单 ----------
        if args.members > 1:
            total = args.members * args.per_account
            print(f"\n组合口径: {args.members} 人 x {args.per_account/1e4:,.0f} 万 "
                  f"= 总权益 {total/1e4:,.0f} 万\n")

            # 负载均衡分配: 按名义金额从大到小, 每次给"加进去后总额最小"的人;
            # 总额相同时优先给净头寸方向相反的人, 让每个人多空自平衡。
            load = [0.0] * args.members
            net = [0.0] * args.members
            member_of = np.zeros(len(rows), dtype=int)
            for i in sorted(range(len(rows)), key=lambda j: -abs(w[j])):
                n = abs(w[i]) * total
                s = 1.0 if w[i] > 0 else -1.0
                best = min(range(args.members),
                           key=lambda m: (load[m] + n, abs(net[m] + s * n)))
                member_of[i] = best
                load[best] += n
                net[best] += s * n

            for r, wi, mi in zip(rows, w, member_of):
                r["member"] = int(mi) + 1
                r["w"] = float(wi)
                r["notional"] = abs(float(wi)) * total
                lots = int(round(r["notional"] / (r["price"] * r["mult"])))
                r["lots"] = lots
                r["side"] = "多" if wi > 0 else "空" if wi < 0 else "—"
                r["notional"] = lots * r["price"] * r["mult"]

            sheet_path = OUT / f"orders_{now:%Y%m%d_%H%M}.csv"
            with sheet_path.open("w", newline="", encoding="utf-8-sig") as f:
                wr = csv.writer(f)
                wr.writerow(["成员", "品种", "方向", "手数", "近月合约", "远月合约",
                             "参考价", "合约乘数", "carry年化", "波动率",
                             "名义金额(元)", "每手保证金", "需保证金(元)"])
                for r in sorted(rows, key=lambda x: (x["member"], -abs(x["carry"]))):
                    wr.writerow([f"成员{r['member']}", r["variety"], r["side"], r["lots"],
                                 r["near"], r["far"], round(r["price"], 2),
                                 int(r["mult"]), f"{r['carry']*100:.1f}%",
                                 f"{r['sigma']*100:.1f}%", round(r["notional"]),
                                 round(r["margin"]), round(r["lots"] * r["margin"])])

            # 每人一块表
            for m in range(1, args.members + 1):
                sub = sorted([r for r in rows if r["member"] == m and r["lots"] > 0],
                             key=lambda x: -abs(x["carry"]))
                if not sub:
                    continue
                gross = sum(r["notional"] for r in sub)
                marg = sum(r["lots"] * r["margin"] for r in sub)
                nl = sum(1 for r in sub if r["side"] == "多")
                ns = sum(1 for r in sub if r["side"] == "空")
                print("=" * 104)
                print(f"【成员 {m}】 {len(sub)} 个品种, 多 {nl} / 空 {ns}, "
                      f"名义 {gross/1e4:,.0f} 万 ({gross/args.per_account:.2f} 倍), "
                      f"需保证金 {marg/1e4:,.1f} 万 / 可用 {args.per_account/1e4:,.0f} 万")
                print("-" * 104)
                print(f"{'品种':<8}{'方向':<6}{'手数':>5}  {'近月':<11}{'远月':<11}"
                      f"{'参考价':>9}{'carry':>9}{'名义万':>8}")
                for r in sub:
                    print(f"{pad_cn(r['variety'].split('.')[-1], 8)}{r['side']:<5}"
                          f"{r['lots']:>5}{r['near'].split('.')[-1]:<11}"
                          f"{r['far'].split('.')[-1]:<11}{r['price']:>9.2f}"
                          f"{r['carry']*100:>8.1f}%{r['notional']/1e4:>8.0f}")
                if marg > 0.9 * args.per_account:
                    print(f"  警告: 该成员保证金占用 {marg/args.per_account:.0%} 过高")

            print("\n" + "=" * 104)
            tot_m = sum(r["lots"] * r["margin"] for r in rows)
            print(f"全组合: 名义 {sum(r['notional'] for r in rows)/1e4:,.0f} 万, "
                  f"保证金合计 {tot_m/1e4:,.0f} 万 "
                  f"({tot_m/total:.0%} 的团队权益)")
            print(f"指令单已导出: {sheet_path}")
            log_row(dict(ts=now.strftime("%Y-%m-%d %H:%M:%S"), mode="sheet",
                         leverage=args.lev, equity=round(total, 2),
                         gross=round(sum(r["notional"] for r in rows), 2),
                         margin=round(tot_m, 2),
                         n_long=sum(1 for r in rows if r["side"] == "多"),
                         n_short=sum(1 for r in rows if r["side"] == "空"),
                         detail=f"members={args.members} file={sheet_path.name}"))
            return 0

        # ---------- 7. 单账户模式(TqKq 排练) ----------
        equity = args.capital if args.capital > 0 else bal
        print(f"\n账户: 权益 {bal:,.0f} 元, 可用 {avail:,.0f} 元")
        if args.capital > 0 and abs(args.capital - bal) > 1:
            print(f"注意: 按下单权益 {args.capital:,.0f} 元算手数(账户实际 {bal:,.0f})")

        targets = []
        for r, wi in zip(rows, w):
            if abs(wi) < 1e-9:
                continue
            lots = int(round(abs(wi) * equity / (r["price"] * r["mult"])))
            if lots <= 0:
                continue
            targets.append(dict(r, w=float(wi), target=lots if wi > 0 else -lots,
                                notional=lots * r["price"] * r["mult"],
                                margin_need=lots * r["margin"]))

        gross = sum(t["notional"] for t in targets)
        need = sum(t["margin_need"] for t in targets)
        print(f"目标敞口 {gross/1e4:,.0f} 万 ({gross/equity:.2f} 倍), "
              f"需保证金 {need/1e4:,.0f} 万, 可用 {avail/1e4:,.0f} 万")
        if need > 0.85 * avail > 0:
            k = 0.85 * avail / need
            print(f"保证金不足, 手数按 {k:.0%} 缩放(避免批量拒单)")
            for t in targets:
                t["target"] = int(t["target"] * k) if abs(t["target"]) * k >= 1 else 0
                t["notional"] *= k
                t["margin_need"] *= k
            targets = [t for t in targets if t["target"] != 0]
            gross = sum(t["notional"] for t in targets)
            need = sum(t["margin_need"] for t in targets)

        targets.sort(key=lambda x: -abs(x["notional"]))
        print(f"\n{'品种':<8}{'近月':<12}{'远月':<12}{'carry':>8}{'波动':>7}"
              f"{'权重':>7}{'手数':>6}{'名义万':>8}")
        print("-" * 100)
        for t in targets:
            print(f"{pad_cn(t['variety'].split('.')[-1], 8)}{t['near'].split('.')[-1]:<12}"
                  f"{t['far'].split('.')[-1]:<12}{t['carry']*100:>7.1f}%{t['sigma']*100:>6.1f}%"
                  f"{t['w']:>7.2f}{t['target']:>6d}{t['notional']/1e4:>8.0f}")

        summary = dict(ts=now.strftime("%Y-%m-%d %H:%M:%S"),
                       mode="live" if args.live else "dry", leverage=args.lev,
                       equity=round(equity, 2), gross=round(gross, 2),
                       margin=round(need, 2),
                       n_long=sum(1 for t in targets if t["target"] > 0),
                       n_short=sum(1 for t in targets if t["target"] < 0),
                       detail=";".join(f"{t['near']}:{t['target']}" for t in targets))

        if not args.live:
            print("\n干跑结束: 没有下任何单。核对后加 --live 才会真的执行。")
            log_row(summary)
            return 0
        if not targets:
            print("目标仓位全为 0, 无需下单")
            log_row(summary)
            return 0

        print("\n开始调整仓位 ...")
        tasks = {t["near"]: TargetPosTask(api, t["near"]) for t in targets}
        for t in targets:
            tasks[t["near"]].set_target_volume(t["target"])
        end = time.time() + 300
        while time.time() < end:
            api.wait_update(deadline=time.time() + 2)
            done = sum(1 for t in targets
                       if int(api.get_position(t["near"]).pos_long)
                       - int(api.get_position(t["near"]).pos_short) == t["target"])
            if done == len(targets):
                break
        miss = [t["near"] for t in targets
                if int(api.get_position(t["near"]).pos_long)
                - int(api.get_position(t["near"]).pos_short) != t["target"]]
        print(f"仓位调整结束, 未达标 {len(miss)}/{len(targets)}"
              f"{' (' + ','.join(miss[:5]) + ' ...)' if miss else ''}")
        log_row(summary)
        print(f"日志已追加: {LOG_CSV}")
        return 0
    finally:
        api.close()


if __name__ == "__main__":
    sys.exit(main())
