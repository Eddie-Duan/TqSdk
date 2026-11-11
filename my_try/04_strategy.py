# -*- coding: utf-8 -*-
"""
体验 4: 一个完整的交易策略（双均线反手 + 仓位管理 + 止损）

运行:  python my_try/04_strategy.py                                        # 基线 10/30
       python my_try/04_strategy.py --fast 5 --slow 20 --exit             # 换参数
       python my_try/04_strategy.py --start 2025-09-01 --end 2026-01-31 --exit   # 换区间

（--exit: 跑完直接退出, 不进浏览器图表; 不加则跑完会停在图表上, 按 Ctrl+C 退出）
（不传参数时, 用的就是下面"参数区"里的默认值）

这是第一个"完整"的策略脚本。策略不是"选个预测算法"就完事, 它由 7 层组成,
下面每一层在代码里都用 ---------- ① ② ③ ... 标了出来:

    ① 数据层   取什么行情
    ② 特征层   从行情算出指标
    ③ 信号层   什么时候进、什么时候出
    ④ 仓位层   下几手
    ⑤ 风控层   亏到什么程度必须走
    ⑥ 执行层   怎么把单子发出去
    ⑦ 评估层   跑完怎么判断好坏

=====================================================================
踩过的两个坑（写在这里, 因为比代码本身更值钱）
=====================================================================

坑 1：绝不能用"还没走完的K线"算信号
    天勤的K线序列里, 最后一根(-1)是正在形成中的K线, 它的 close 只是"当前价",
    会一直变。如果用它算均线并判断交叉, 会和下一根K线看到的"上一根"对不上,
    信号就会被吞掉 —— 实测真的漏掉了一次死叉。
    正确做法: 判断信号时排除最后一根, 只用已收盘的K线（代码里的 c = close[:-1]）。

坑 2：主连合约的K线, 在回测里未必有足够长的历史
    天勤官方推荐"信号用主连 KQ.m@..., 交易用 quote.underlying_symbol"。
    但实测: 回测里 KQ.m@SHFE.rb 的K线只从"最近一次换月之后"才有数据
    （rb 最近一次换月是 2026-04-08）, 结果回测前两个月完全没数据、策略空转。
    （静态换月表文件本身是完整的, 所以问题出在K线数据上。）
    因此改成: 信号和交易都用同一个真实合约 —— 这也符合官方规范里
    "下单/持仓/TargetPosTask/保证金 要用真实未下市合月合约"的要求。
    代价: 该合约成为主力之前流动性差, 回测成交价偏乐观, 写报告时要说明。

跑完会在 my_try/output/ 下导出两个 CSV（文件名带参数后缀, 不会互相覆盖）。
"""
import csv
import os
from datetime import date, datetime, timedelta, timezone

from tqsdk import TqApi, TqAuth, TqBacktest, TqSim, TargetPosTask, BacktestFinished

from tq_auth import TQ_USER, TQ_PASS

# ==================== 参数区（改这里就行） ====================
SYMBOL = "SHFE.rb2610"          # 交易合约: 信号和下单都用它
START = date(2026, 2, 1)        # 回测开始日期
END = date(2026, 6, 30)         # 回测结束日期
DURATION = 60 * 60              # K线周期(秒), 3600 = 1 小时线
MA_FAST, MA_SLOW = 10, 30       # 快线 / 慢线周期
INIT_BALANCE = 100_000          # 初始资金
POSITION_PCT = 0.30             # ④ 单次开仓最多用多少比例的权益做保证金
MARGIN_RATE = 0.12              # ④ 假设的保证金率（天勤不对外提供保证金率, 这里用假设值）
STOP_LOSS_PCT = 0.02            # ⑤ 单笔止损: 相对开仓价逆向 2%
KEEP_WEB_CHART_OPEN = True      # 跑完是否保留程序以查看浏览器里的回测图表
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# ==================== 命令行参数（可选, 用来做参数对比实验） ====================
#   python my_try/04_strategy.py --fast 5 --slow 20 --exit
# 不传参数时用的就是上面"参数区"里的默认值。
import argparse

_parser = argparse.ArgumentParser(description="双均线反手策略回测")
_parser.add_argument("--fast", type=int, default=MA_FAST, help=f"快线周期 (默认 {MA_FAST})")
_parser.add_argument("--slow", type=int, default=MA_SLOW, help=f"慢线周期 (默认 {MA_SLOW})")
_parser.add_argument("--stop", type=float, default=STOP_LOSS_PCT, help=f"止损比例 (默认 {STOP_LOSS_PCT})")
_parser.add_argument("--start", default=START.isoformat(), help=f"回测开始日期 (默认 {START})")
_parser.add_argument("--end", default=END.isoformat(), help=f"回测结束日期 (默认 {END})")
_parser.add_argument("--exit", action="store_true", help="跑完立即退出, 不保留浏览器回测图表")
_args = _parser.parse_args()

MA_FAST, MA_SLOW, STOP_LOSS_PCT = _args.fast, _args.slow, _args.stop
START, END = date.fromisoformat(_args.start), date.fromisoformat(_args.end)
if _args.exit:
    KEEP_WEB_CHART_OPEN = False

if not (0 < MA_FAST < MA_SLOW):
    raise SystemExit(f"参数错误: 需要 0 < 快线({MA_FAST}) < 慢线({MA_SLOW}), 请检查 --fast / --slow")
if START >= END:
    raise SystemExit(f"参数错误: 开始日期({START})必须早于结束日期({END})")

# 输出文件名带上参数和区间, 不同实验互不覆盖
TAG = f"{MA_FAST}_{MA_SLOW}_{START:%Y%m%d}_{END:%Y%m%d}"


def _is_num(x):
    """排除 None 和 nan（行情没到的时候字段是 nan）"""
    return x is not None and x == x


def fmt_dt(ns):
    """天勤的K线时间是从 epoch 起的纳秒数, 这里换算成北京时间(UTC+8)显示"""
    if not _is_num(ns):
        return ""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M")


def estimate_volume(equity, price, volume_multiple):
    """④ 仓位层: 手数 = 权益 × 仓位比例 ÷ 每手保证金"""
    if not (_is_num(price) and price > 0 and _is_num(volume_multiple) and volume_multiple > 0):
        return 1
    lot_margin = price * volume_multiple * MARGIN_RATE     # 每手保证金
    return max(1, int(equity * POSITION_PCT / lot_margin))


acc = TqSim(INIT_BALANCE)
api = TqApi(acc, backtest=TqBacktest(start_dt=START, end_dt=END),
            auth=TqAuth(TQ_USER, TQ_PASS))

# ---------- ① 数据层 ----------
# get_* 返回的是"引用对象", 不是快照; 数据会在每次 wait_update() 时自动刷新
klines = api.get_kline_serial(SYMBOL, DURATION, data_length=MA_SLOW + 3)
quote = api.get_quote(SYMBOL)
account = api.get_account()
    
# ---------- ⑥ 执行层: 调仓工具, 声明一次即可 ----------
target_pos = TargetPosTask(api, SYMBOL)

# 交易状态
my_volume = 1           # 最近一次信号算出的手数
my_target = 0           # 想要的目标持仓: 正=多, 负=空, 0=空仓
stopped = False         # 止损后置 True, 要等出现新的交叉信号才解除

bar_log, signal_log = [], []
first_bar_printed = False

print(f"回测开始 {START} ~ {END}   合约 {SYMBOL}   初始资金 {INIT_BALANCE:,}")
print(f"参数: 均线 {MA_FAST}/{MA_SLOW}, 止损 {STOP_LOSS_PCT * 100:.1f}%, 单次仓位上限 {POSITION_PCT * 100:.0f}%\n")

try:
    while True:
        api.wait_update()

        # 只在产生新K线时做判断（否则每个 tick 都会触发一次）
        # 注意: 新K线出现的这一刻, 上一根K线才刚刚收盘
        if not api.is_changing(klines.iloc[-1], "datetime"):
            continue

        close_all = klines.close
        if len(close_all) < MA_SLOW + 3 or not _is_num(close_all.iloc[-2]):
            continue

        # ---------- 关键: 丢掉最后一根"未完成"的K线, 只用已收盘的 ----------
        close = close_all.iloc[:-1]
        bar_dt = fmt_dt(klines.datetime.iloc[-2])     # 刚收盘那根K线的时间
        px = close.iloc[-1]                            # 刚收盘那根K线的收盘价

        if not first_bar_printed:
            print(f"[首根已收盘K线] {bar_dt}  收盘 {px}\n")
            first_bar_printed = True

        # ---------- ② 特征层: 均线（同样只用已收盘K线）----------
        fast = close.iloc[-MA_FAST:].mean()
        slow = close.iloc[-MA_SLOW:].mean()
        fast_prev = close.iloc[-MA_FAST - 1:-1].mean()
        slow_prev = close.iloc[-MA_SLOW - 1:-1].mean()

        # ---------- ③ 信号层: 金叉做多 / 死叉做空（信号必须同时定义"进"和"出"）----------
        golden = fast_prev <= slow_prev and fast > slow      # 金叉
        dead = fast_prev >= slow_prev and fast < slow        # 死叉

        u_price = quote.last_price

        if golden:
            stopped = False
            my_volume = estimate_volume(account.balance, u_price, quote.volume_multiple)
            my_target = my_volume
            print(f"{bar_dt} [金叉] 做多 {my_volume} 手  收盘价 {px}")
            signal_log.append((bar_dt, "金叉做多", SYMBOL, px, my_target))
        elif dead:
            stopped = False
            my_volume = estimate_volume(account.balance, u_price, quote.volume_multiple)
            my_target = -my_volume
            print(f"{bar_dt} [死叉] 做空 {my_volume} 手  收盘价 {px}")
            signal_log.append((bar_dt, "死叉做空", SYMBOL, px, my_target))

        # ---------- ⑤ 风控层: 止损, 优先级高于信号 ----------
        position = api.get_position(SYMBOL)
        if not stopped and _is_num(u_price):
            if position.pos_long > 0 and _is_num(position.open_price_long) and position.open_price_long > 0:
                ret = (u_price - position.open_price_long) / position.open_price_long
                if ret <= -STOP_LOSS_PCT:
                    stopped, my_target = True, 0
                    print(f"{bar_dt} [止损] 多头浮亏 {ret * 100:.2f}%, 强制平仓")
                    signal_log.append((bar_dt, f"止损平多 {ret * 100:.2f}%", SYMBOL, u_price, 0))
            elif position.pos_short > 0 and _is_num(position.open_price_short) and position.open_price_short > 0:
                ret = (position.open_price_short - u_price) / position.open_price_short
                if ret <= -STOP_LOSS_PCT:
                    stopped, my_target = True, 0
                    print(f"{bar_dt} [止损] 空头浮亏 {ret * 100:.2f}%, 强制平仓")
                    signal_log.append((bar_dt, f"止损平空 {ret * 100:.2f}%", SYMBOL, u_price, 0))

        # ---------- ⑥ 执行层: 只声明目标持仓, 拆单/追价/撤单重发都交给 TargetPosTask ----------
        target_pos.set_target_volume(my_target)

        # ---------- ⑦ 评估层: 每根K线记一笔, 后面用来画图和统计 ----------
        bar_log.append((bar_dt, px, round(fast, 2), round(slow, 2),
                        position.pos_long, position.pos_short,
                        round(account.balance, 2), my_target))

except BacktestFinished:
    stat = acc.tqsdk_stat

    print("\n================ 回测结果 ================")
    print(f"策略: 双均线({MA_FAST}/{MA_SLOW}) 反手 + 止损 {STOP_LOSS_PCT * 100:.1f}%")
    print(f"合约: {SYMBOL}   区间: {START} ~ {END}")
    print(f"初始资金:   {INIT_BALANCE:,.2f}")
    print(f"最终权益:   {account.balance:,.2f}")
    print(f"收益率:     {stat['ror'] * 100:.2f}%")
    print(f"年化收益率: {stat['annual_yield'] * 100:.2f}%")
    print(f"最大回撤:   {stat['max_drawdown'] * 100:.2f}%")
    print(f"年化夏普:   {stat['sharpe_ratio']:.2f}")
    print(f"胜率:       {stat['winning_rate'] * 100:.2f}%")
    print(f"总手续费:   {stat['commission']:,.2f}")

    # 自己按权益曲线算一遍最大回撤, 和上面的数字对照（学会自己验证, 别只信黑箱）
    if bar_log:
        peak, mdd = -float("inf"), 0.0
        for row in bar_log:
            peak = max(peak, row[6])
            if peak > 0:
                mdd = max(mdd, (peak - row[6]) / peak)
        print(f"自行计算的最大回撤: {mdd * 100:.2f}%   (用于对照)")

    # ---------- 导出 CSV（utf-8-sig: Excel 打开不会乱码）----------
    os.makedirs(OUT_DIR, exist_ok=True)
    bars_path = os.path.join(OUT_DIR, f"bars_{TAG}.csv")
    sig_path = os.path.join(OUT_DIR, f"signals_{TAG}.csv")

    with open(bars_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["K线时间", "收盘价", f"MA{MA_FAST}", f"MA{MA_SLOW}",
                    "多头手数", "空头手数", "账户权益", "目标持仓"])
        w.writerows(bar_log)

    with open(sig_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["时间", "事件", "合约", "价格", "目标持仓"])
        w.writerows(signal_log)

    print(f"\n已导出 {len(bar_log)} 根K线、{len(signal_log)} 个信号:")
    print(f"  {bars_path}")
    print(f"  {sig_path}")
    print(f"用 Excel 打开 {os.path.basename(bars_path)}, 选中『账户权益』列插入折线图, 就是权益曲线。")

    # 一行摘要, 方便把不同参数的结果贴在一起对比
    print(f"[SUMMARY] fast={MA_FAST} slow={MA_SLOW} stop={STOP_LOSS_PCT} "
          f"ror={stat['ror']:.4f} mdd={stat['max_drawdown']:.4f} "
          f"sharpe={stat['sharpe_ratio']:.3f} commission={stat['commission']:.2f} "
          f"bars={len(bar_log)} events={len(signal_log)}")

    if KEEP_WEB_CHART_OPEN:
        print("\n浏览器里已打开回测报告图表。按 Ctrl+C 结束程序。")
        try:
            while True:
                api.wait_update()
        except KeyboardInterrupt:
            pass

    api.close()