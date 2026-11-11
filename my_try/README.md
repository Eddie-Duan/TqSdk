# my_try —— TqSdk 体验脚手架

一组由浅入深的脚本，用来体验 TqSdk 并做策略研究。**它们不是项目的一部分，随时可以整个文件夹删掉。**

## 使用步骤

### 1. 注册快期账户（免费，必须）

https://account.shinnytech.com/

登录名可以用 **手机号 / 用户名 / 邮箱** 任意一个，密码就是待会儿要填的那个。

### 2. 填账号密码

打开 `tq_auth.py`，把这两行改成你自己的：

```python
TQ_USER = "13800000000"
TQ_PASS = "你的密码"
```

> 这个文件已经写进同目录的 `.gitignore`，不会被提交到 git。

### 3. 依次运行

在项目根目录 `TqSdk\` 下执行（终端里直接敲）：

| 脚本 | 功能 | 备注 |
|---|---|---|
| `python my_try/01_quote.py` | 实时行情 | 打印 20 秒自动退出；收盘时段也能跑，价格只是不跳 |
| `python my_try/02_backtest.py` | 双均线策略回测 | 回测次数不限；结束时会自动开浏览器显示回测图表，按 `Ctrl+C` 退出 |
| `python my_try/03_sim_trade.py` | 模拟交易下单 | 用长期有效的快期模拟账户，持仓能在快期 APP 里看到；建议交易时段跑 |
| `python my_try/04_strategy.py` | **完整策略**（双均线反手 + 仓位管理 + 止损） | 把策略的 7 个层显式拆开；跑完在 `my_try/output/` 导出两个 CSV 给报告用 |
| `python my_try/07_hist_range.py` | **数据体检**：主连/指数/真实合约各能回溯多长历史 | 只读数据，可反复跑 |
| `python my_try/08_robust_test.py` | **稳健性检验**：4.75 年 × 8 组参数，输出逐年收益矩阵 + 参数热力图 | 只读数据；用指数合约 `KQ.i@` 做长历史信号研究，成交是简化假设 |
| `python my_try/09_variance_ratio.py` | **方差比检验**：测量趋势存在于哪个时间尺度 | 只读数据；用来交叉验证 `08` 的参数热力图 |
| `python my_try/10_regime_filter.py` | **regime 过滤**：滚动 VR 可预测性 + 当成交易规则实测 | 只读数据；结论是"事后可见、事前不可用" |
| `python my_try/11_basis_test.py` | **期限结构/基差**：价差是否比价格更均值回归 | 只读数据；换到"有理论锚"的第 3 层信息源 |
| `python my_try/12_basis_long.py` | **长样本检验**：按换月表拼接"主力−次主力"合约对，把样本从 0.67 年扩到 4 年 | 只读数据；用来验证 `11` 的信号是不是只在那几个月成立 |

## 想自己改点什么

| 想改的 | 改哪里 |
|---|---|
| 交易品种 | 脚本顶部的 `SYMBOL`，如 `SHFE.cu2610`(铜)、`DCE.i2701`(铁矿)、`KQ.m@SHFE.rb`(螺纹主连) |
| 账户类型 | `TqSim()` 本地临时模拟 / `TqKq()` 长期模拟 / `TqAccount()` 实盘（需专业版） |
| 策略参数 | `02_backtest.py` 里的 `SHORT` / `LONG` / `DURATION` / `START` / `END` |
| 参考更多策略 | `tqsdk/demo/example/` 下有 40 多个完整示例 |

## 环境说明（重要）

本仓库已用 `pip install -e .` 装进你的 Python 3.14，但**必须**把 `shinny-structlog` 钉在 0.0.4：

```powershell
pip install "shinny-structlog==0.0.4"
```

原因：`shinny-structlog` 0.0.5（2026-09-16 发布）是破坏性更新，删掉了全局接管 `logging.Logger` 的行为，
导致 TqSdk 3.10.2 一启动就报 `TypeError: Logger._log() got an unexpected keyword argument 'product'`。
如果以后重新装过 tqsdk 又出现这个错，重跑上面这条命令即可。

## 常见坑

- **必须联网**，且必须有快期账户，没有账号跑不通。
- 首次运行时订阅行情可能瞬时超时（冷启动），重跑一次通常就好。
- 想看内部调试日志，把 `TqApi(..., debug=True)` 加上，日志在 `~\.tqsdk\logs\`。
- 回测里**不能直接交易主连合约**（`KQ.m@...`），要用 `quote.underlying_symbol` 拿到当期主力合约再下单。
- 免费版限制：实盘交易属专业版功能；K 线每个序列最多取 8000 根（1 小时线约 4.75 年，日线约 10 年）。
