# -*- coding: utf-8 -*-
"""
体验 2: 策略回测(双均线) —— 在历史行情上重放策略。

运行:  python my_try/02_backtest.py

说明:
  * 回测不用等开市, 也不用真钱, TqSdk 会用内置的 TqSim 模拟账户撮合。
  * 回测次数不限, 可以随意重复跑。
  * 结束时打印统计信息, 并自动在浏览器打开一张回测报告图表, 按 Ctrl+C 退出。
"""
from datetime import date

from tqsdk import TqApi, TqAuth, TqBacktest, TqSim, TargetPosTask, BacktestFinished

from tq_auth import TQ_USER, TQ_PASS

SYMBOL = "SHFE.rb2701"      # 回测标的: 螺纹钢 2027年1月 合约
START = date(2026, 2, 1)    # 回测开始日期
END = date(2026, 6, 30)     # 回测结束日期
DURATION = 60 * 60          # K线周期(秒): 1 小时线
SHORT, LONG = 10, 30        # 快线、慢线周期
INIT_BALANCE = 100_000      # 初始资金

acc = TqSim(INIT_BALANCE)
api = TqApi(acc, backtest=TqBacktest(start_dt=START, end_dt=END),
            auth=TqAuth(TQ_USER, TQ_PASS))

account = api.get_account()
klines = api.get_kline_serial(SYMBOL, DURATION, data_length=LONG + 2)
target_pos = TargetPosTask(api, SYMBOL)

print(f"回测开始: {START} ~ {END}   初始资金 {INIT_BALANCE}   标的 {SYMBOL}\n")

try:
    while True:
        api.wait_update()
        # 只在新K线产生时判断, 否则每个 tick 都会触发一次
        if api.is_changing(klines.iloc[-1], "datetime"):
            close = klines.close
            short_ma = close.iloc[-SHORT:].mean()
            long_ma = close.iloc[-LONG:].mean()
            if short_ma > long_ma:
                target_pos.set_target_volume(1)     # 快线在上: 目标持有多头 1 手
            else:
                target_pos.set_target_volume(0)     # 否则: 目标空仓
except BacktestFinished:
    stat = acc.tqsdk_stat

    print("\n================ 回测结果 ================")
    print(f"初始资金:   {INIT_BALANCE:,.2f}")
    print(f"最终权益:   {account.balance:,.2f}")
    print(f"收益率:     {stat['ror'] * 100:.2f}%")
    print(f"年化收益率: {stat['annual_yield'] * 100:.2f}%")
    print(f"最大回撤:   {stat['max_drawdown'] * 100:.2f}%")
    print(f"年化夏普:   {stat['sharpe_ratio']:.2f}")
    print(f"胜率:       {stat['winning_rate'] * 100:.2f}%")
    print(f"总手续费:   {stat['commission']:,.2f}")
    print("=========================================")

    # 让程序保持运行, 浏览器里才能看到回测报告图表; 按 Ctrl+C 退出
    print("\n浏览器已打开回测报告图表。按 Ctrl+C 结束程序。")
    try:
        while True:
            api.wait_update()
    except KeyboardInterrupt:
        pass
    api.close()
