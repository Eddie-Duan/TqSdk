# -*- coding: utf-8 -*-
"""
体验 3: 模拟交易 —— 用快期模拟账户真实下单(花的都是模拟资金)。

运行:  python my_try/03_sim_trade.py

说明:
  * TqKq() 是一个长期有效的快期模拟账户, 持仓/委托/成交记录会保存在服务器上,
    重启程序不会丢, 还能在"快期"手机 APP 里直接看到。
  * 建议在交易时段运行, 非交易时段委托会挂着等开盘。
  * 想平仓, 把 VOLUME 改成 0 再跑一次即可。
"""
import time

from tqsdk import TqApi, TqAuth, TqKq, TargetPosTask

from tq_auth import TQ_USER, TQ_PASS

SYMBOL = "SHFE.rb2701"      # 交易标的
VOLUME = 1                  # 目标持仓手数: 正数=多头, 负数=空头, 0=空仓

api = TqApi(TqKq(), auth=TqAuth(TQ_USER, TQ_PASS))

account = api.get_account()
quote = api.get_quote(SYMBOL)
position = api.get_position(SYMBOL)
target_pos = TargetPosTask(api, SYMBOL)

print("---------- 下单前 ----------")
print(f"账户权益: {account.balance:,.2f}   可用资金: {account.available:,.2f}")
print(f"当前持仓: 多 {position.pos_long} 手 / 空 {position.pos_short} 手")

print(f"\n设置目标持仓 {VOLUME} 手, 等待成交 ...")
target_pos.set_target_volume(VOLUME)

deadline = time.time() + 30
while time.time() < deadline:
    api.wait_update(deadline=deadline)
    if api.is_changing(position) or api.is_changing(account, "balance"):
        print(f"{quote.datetime}  最新价 {quote.last_price}  "
              f"持仓 多{position.pos_long}/空{position.pos_short}  "
              f"浮动盈亏 {account.float_profit:,.2f}  风险度 {account.risk_ratio * 100:.2f}%")

print("\n---------- 下单后 ----------")
print(f"账户权益: {account.balance:,.2f}   可用资金: {account.available:,.2f}")
print(f"当前持仓: 多 {position.pos_long} 手 / 空 {position.pos_short} 手")
print(f"持仓均价(多): {position.open_price_long}   浮动盈亏: {account.float_profit:,.2f}")

api.close()
print("\n已断开连接。打开快期 APP 可以看到这笔模拟持仓。")
