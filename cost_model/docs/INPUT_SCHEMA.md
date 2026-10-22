# 输入、完整计划与输出约定

## 紧凑配置

`predict` 的输入是一个 JSON 对象或对象列表。

| 字段 | 类型/含义 |
|---|---|
| `implementation` | `catlass`、`base`、`scsk`、`deterministic`；README中的ops_nn别名也可用 |
| `shape` | 三个整数 `[M,N,K]`；数学意义是A[M,K]乘B[K,N] |
| `dtype` | `float16`/`fp16` 或 `bfloat16`/`bf16`，默认float16 |
| `cores` | 明确的启动核数，整数1..24，DET另有非空分片限制 |
| `tiling` | 以下两种实现相关格式之一，不能省略后让模型猜测 |
| `case_id` | 可选的非空字符串，用于与实测关联；不参与耗时计算 |
| `trans_a`,`trans_b` | 可省略；若写出必须是JSON boolean false |
| `batch` | 可省略；若写出只能是整数1 |
| `execution_profile` | 可省略；只能是`warm400_gate_targetC` |

CATLASS的`tiling`格式：

```json
{"preset": "deepk"}
```

或者明确给出两个tile，顺序都是M、N、K：

```json
{"l1_tile": [16,16,4096], "l0_tile": [16,16,1024]}
```

两种方式不能混用并提供矛盾值。数值方式仍须匹配已经审查的四组L1/L0，不表示支持任意tile。

ops-nn的`tiling`格式：

```json
{"base_m":128, "base_n":128, "base_k":128, "step_k":2}
```

`step_k`是K方向的L1 step，补全时同时设置`stepKa=stepKb`。一个L1 K包的静态跨度为`baseK*stepK`，不能把`step_k`当成K维度或Split-K核数。Split-K核数由`cores`及实际任务分配共同决定。

未知字段会报错，帮助发现`baseM`/`base_m`等拼写和单位误用。`measured_us`、`time_ms`、PMU和目标计时不是预测输入。若手头是本程序的完整预测对象，也可传给`predict`，其`static_input`会被明确提取用于复算。

## ops-nn计划如何补全

以baseM=bm、baseN=bn、baseK=128、stepK=s为例：

| 项目 | BASE | 核内Split-K | 确定性Split-K |
|---|---|---|---|
| singleCoreM | bm | bm | 384 |
| singleCoreN | bn | bn | 输入N |
| singleCoreK | 输入K | min(K,128s) | min(K,128s) |
| stepM / stepN | 1 / 1 | 1 / 1 | 3 / 1 |
| depthA1 / depthB1 | 2s / 2s | 2s / 2s | 3s / 2s |
| dbL0A / dbL0B | 2 / 2 | 2 / 2 | 2 / 2 |
| dbL0C | 1 | 1 | 2 |
| iterateOrder | 1 | 1 | 1 |

其余字段包含shareL1Size、shareL0CSize、布局转换条件、任务网格及70个raw_words，均从同一静态规则构造。转换标志按已核对路径计算，不能只用FLOPs解释N16与N24的耗时差别。

这些值描述本次已验证的显式实现，不是对所有ops-nn/CANN版本的TCubeTiling通用编码器。原生host tiler如果选择了不同计划或kernel分支，应将其实际语义和计划纳入新的验证。

## 完整计划输入

`examples/full_plan_deterministic.json`由同一个`expand_config`产生，可直接预测。结构为：

```text
implementation, shape, dtype, cores, case_id
trans_a=false, trans_b=false, batch=1
execution_profile="warm400_gate_targetC"
plan:
  m,n,k,dtype,trans_a,trans_b
  tcube: usedCoreNum,M,N,Ka,Kb,singleCoreM/N/K,
         baseM/N/K,stepM/N,stepKa/Kb,depthA1/B1,dbL0A/B/C,
         iterateOrder,shareUbSize,shareL1Size,shareL0CSize
  raw_words: 70个ABI整数
  nd2nz_a,nd2nz_b,block_dim
```

不能同时提供`tiling`和`plan`，也不能用CATLASS字段描述ops-nn。top-level dtype与plan dtype必须一致；完整计划中使用规范`float16`或`bfloat16`。所有tcube和raw_words值必须是整数。检查不仅比较baseM/baseN，也比较全部缓冲/步长、原始words及转换标志，避免忽略影响实现行为的配置。

CATLASS完整输入使用`tile`、`l1_tile`、`l0_tile`替代`tiling`，没有`plan`。

当前额外源偏移只能为0；没有任意stride/offset/transport覆写入口。布局、双缓冲与槽释放的影响已经通过静态计划和固定模型进入计算，但这不等于任意手改布局/缓冲都已校准。

## 预测输出

| 字段 | 含义 |
|---|---|
| `predicted_us` / `predicted_ms` | 同一个预测设备kernel时间，分别为微秒和毫秒 |
| `cores` | 输入的启动核数 |
| `active_compute_cores` | 源码任务分配后实际承担GEMM计算的核数；不等于转换/归约参与者总数 |
| `formula` | 默认`fixed_calibrated`；semantic模式为`semantic_pipeline` |
| `static_input` | 经过检查并补全的可复算计划 |
| `model_version`,`hardware` | 固定模型和校准硬件 |
| `execution_assumption` | 冻结预热协议，不表示预测端执行了预热 |
| `target_measurements_required` | 固定false |
| `hardware_execution_performed` | 固定false |
| `unseen_input_error_guaranteed` | false，未见输入没有逐点数学误差保证 |
| `all_candidate_orders_certified` | false，完整候选排序仍有独立反例 |

`--explain`附加`formula_details`和`semantic_pipeline`，保留实际公式计算出的预算、修正和关键路径账目。内部历史字段名用于来源追溯；发布资格以`model_info.json`和README的独立成绩为准，不能把某个内部旧版本标志理解为新输入已经实测。

## 实测CSV

字段是`case_id,measured_us`，建议加`process_index`。每一行是一个正式进程的批量计时聚合值；至少三个独立进程才能统计stable5。模型结果与CSV的case_id集合必须完全一致，重复case_id/process_index会报错。

`MAPE=mean(abs(predicted/median_process(measured)-1))`。最大APE是某个配置的误差，严格错序和stable5只在同实现、同输入、同dtype内部比较。允许真实近差/波动并列，仍保留严格顺序统计；实测稳定差距超过5%的预测并列会单独记为prediction_ties。
