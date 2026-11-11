# -*- coding: utf-8 -*-
"""
体验 1: 实时行情 —— 先确认账号和行情连接都通了。

运行:  python my_try/01_quote.py

说明:
  * 程序会订阅合约行情, 打印 20 秒后自动退出, 不会一直卡住。
  * 收盘时段也能跑, 只是价格不再跳动。
  * 想换品种, 改下面的 SYMBOL 即可; 主力连续合约写法是 "KQ.m@SHFE.rb"。
"""
import time

from tqsdk import TqApi, TqAuth

from tq_auth import TQ_USER, TQ_PASS

SYMBOL = "SHFE.rb2701"      # 上期所 螺纹钢 2027年1月 合约

api = TqApi(auth=TqAuth(TQ_USER, TQ_PASS))
quote = api.get_quote(SYMBOL)

print(f"合约: {quote.instrument_id}  {quote.instrument_name}")
print(f"最小变动价位: {quote.price_tick}")
print("开始接收行情, 20 秒后自动退出 ...\n")

deadline = time.time() + 20
while time.time() < deadline:
    # deadline 到了还没新数据就返回 False, 不会抛异常
    api.wait_update(deadline=deadline)
    print(f"{quote.datetime}  最新价 {quote.last_price}  "
          f"买一 {quote.bid_price1}/{quote.bid_volume1}  "
          f"卖一 {quote.ask_price1}/{quote.ask_volume1}  "
          f"成交量 {quote.volume}  持仓量 {quote.open_interest}")

api.close()
print("\n已断开连接。")
