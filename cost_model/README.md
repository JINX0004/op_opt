# Matmul 静态性能模型：独立核心代码

**推理只需要 Python 3.10 或更新版本和本目录，不需要 NPU、CANN、CATLASS/ops-nn 源码、NumPy、PyTorch、联网或目标 shape 的计时/PMU。** Windows、Linux 上使用相同的纯 Python 入口。

输入实现方式、`M/N/K`、dtype、明确的 tiling 和启动核数，输出预测的**完整 matmul kernel 耗时**，单位为微秒 `predicted_us`。模型参数在离线阶段已经冻结；使用时不重新拟合。

当前模型为 `NN_component_service_v3490_BASE64`，硬件校准目标为 **Ascend 910B2**。这不是任意硬件、任意 GEMM 实现的通用预测器。先读下方的支持范围；不支持的参数会被拒绝，不能把未知配置静默套用到相似模板。

## 1. 目录内容

```text
cost_model_matmul_corecode/
  README.md                         本说明
  pyproject.toml                    可选的 Python 安装配置
  matmul_cost_model/
    api.py                          配置检查、预测、候选生成与比较
    cli.py                          命令行入口
    evaluation.py                   与外部实测 CSV 比较误差/排序
    data/model.json                 最新冻结参数，保持原始文件与 SHA256
    data/model_info.json            支持范围、独立成绩与版本
    data/source_provenance.json     源文件提取和命名空间调整记录
    _engine/                        实际必需的静态公式/流水递推依赖
  examples/                         紧凑输入、完整计划、API 和真实实测比较示例
  tests/                            单元测试和压缩的独立验证样本
  tools/verify_release.py           完整性、自测、1198 配置复算
  tools/benchmark_cpu.py            模型自身的 CPU 评分开销测试
  docs/INPUT_SCHEMA.md              输入字段和完整 tiling 说明
  docs/MODEL_AND_SCOPE.md           公式、测量口径和支持边界
  MANIFEST.json                     交付文件 SHA256 清单
```

`_engine` 中一些文件带有历史版本号。这些是**最新模型实际调用的公式层和递推依赖**，不是多个备用模型，不能按文件名的新旧随意删除。没有打包原项目的实验日志、profiling 大文件、训练脚本、NPU 二进制或 SSH 配置。历史参数层嵌入在固定 `model.json` 中，预测需要保留。

## 2. 最快开始：无需安装

把目录解压到任意位置，打开终端并进入目录：

```bash
cd cost_model_matmul_corecode
python --version
python -B -m matmul_cost_model info --output results/model_info.json
python -B -m matmul_cost_model predict --input examples/fp16_deterministic.json --output results/prediction.json
```

`results/` 在指定输出时自动创建；不指定 `--output` 时 JSON 输出到终端。预测不会调用设备、SSH 或 profiling 工具，也不会创建测量任务。

这个示例是 FP16、NN、`M/N/K=16/16/102400`、ops-nn 确定性 Split-K、`base128×128×128, stepK=2, cores=24`。结果中的主要字段为：

```json
{
  "implementation": "deterministic",
  "shape": [16, 16, 102400],
  "dtype": "float16",
  "cores": 24,
  "active_compute_cores": 24,
  "predicted_us": 91.98508811412263,
  "predicted_ms": 0.09198508811412263,
  "target_measurements_required": false,
  "hardware_execution_performed": false
}
```

启动核数与实际有效计算核数分别输出。例如这个输入使用 BASE、32×64×128/step4、启动8核时，只有1个有效计算核，预测约851.536us；不能把总工作简单除以8。

也可以可选安装为 Python 包：

```bash
python -m pip install --no-build-isolation .
matmul-cost-model info
```

可选安装需要安装环境中已有 setuptools 和 wheel 构建工具；**直接在解压目录运行不需要这些工具，也不需要 pip 安装任何运行依赖**。在别的项目中长期调用可安装此包，或把本目录加入该项目的 Python 模块搜索路径。

## 3. 四种实现的名称和含义

| implementation | 含义 | 需要的 tiling |
|---|---|---|
| `catlass` | 当前已校准的 CATLASS 09 Split-K 路径 | L1/L0 tile 或已审查 preset |
| `base` | ops-nn Matmul BASE，MN 分块、任务内遍历 K | baseM/baseN/baseK、stepK |
| `scsk` | ops-nn 核内 Split-K，每个输出任务在核内沿 K 分段计算/更新 | baseM/baseN/baseK、stepK |
| `deterministic` | ops-nn 跨核确定性 Split-K，分配 K 任务后归约输出 | baseM/baseN/baseK、stepK |

别名也可使用：`ops_nn_base`、`ops_nn_single_core_split_k`、`ops_nn_deterministic_split_k`。`scsk` 的“核内”描述每个任务的 K 分段方式，并不表示所有 MN 输入始终只启动一个核。

dtype 支持 `fp16`/`float16`、`bf16`/`bfloat16`，输出统一使用后两种规范名称。

## 4. 指定 tiling 的命令行预测

### CATLASS：指定 L1/L0 preset

```bash
python -B -m matmul_cost_model predict --shape 16 16 102400 --dtype fp16 --implementation catlass --tile deepk --cores 16
```

`deepk` 对应 L1 `[16,16,4096]`、L0 `[16,16,1024]`，该例预测约 **98.694us**。直接用 tile 数值的 JSON 见 `examples/catlass_explicit.json`。

### ops-nn：指定基础块和 K step

```bash
python -B -m matmul_cost_model predict --shape 16 16 102400 --dtype fp16 --implementation base --base-m 32 --base-n 64 --base-k 128 --step-k 4 --cores 8
python -B -m matmul_cost_model predict --shape 16 16 102400 --dtype fp16 --implementation scsk --base-m 128 --base-n 128 --base-k 128 --step-k 4 --cores 16
python -B -m matmul_cost_model predict --shape 16 16 102400 --dtype fp16 --implementation deterministic --base-m 128 --base-n 128 --base-k 128 --step-k 2 --cores 24
python -B -m matmul_cost_model predict --shape 64 128 512 --dtype bf16 --implementation base --base-m 128 --base-n 128 --step-k 4 --cores 1
```

四条命令预测约为 **851.536us、845.330us、91.985us、4.705us**。命令行省略 `--base-k` 时默认128，其余 tiling 维度必须明确给出。

这几组值的完整配置已经放在 `examples/four_paths_fp16.json`、`examples/four_paths_bf16.json` 中。一次预测整个列表：

```bash
python -B -m matmul_cost_model predict --input examples/four_paths_fp16.json --output results/four_paths_fp16.json
python -B -m matmul_cost_model predict --input examples/four_paths_bf16.json --output results/four_paths_bf16.json
```

## 5. 使用 JSON 或完整 TCubeTiling 计划

推荐的紧凑输入：

```json
{
  "case_id": "my_det24",
  "implementation": "deterministic",
  "shape": [16, 16, 102400],
  "dtype": "float16",
  "cores": 24,
  "tiling": {"base_m": 128, "base_n": 128, "base_k": 128, "step_k": 2}
}
```

模型按已经核对的静态策略补全 singleCoreM/N/K、stepM/N、L1/L0 缓冲、ND→NZ 标志和70个 ABI words。补全是**固定的计划构造规则**，不查询实测表，也不运行 CANN host tiler。

若已有完整计划，使用 `examples/full_plan_deterministic.json` 的格式。模型会检查 `plan.tcube`、`raw_words`、布局、shape、dtype、block_dim 等是否一致。任意更改 depth、dbL0、step、stride 或转换标志，不会被静默忽略后给出看似精确的耗时。当前只接受与已审查模板一致的完整计划。

详情见 [docs/INPUT_SCHEMA.md](docs/INPUT_SCHEMA.md)。`case_id` 仅用于结果关联，不影响预测；省略时自动由静态配置生成。不要把 `measured_us`、PMU 或真实计时放入预测输入。

## 6. 同一输入下比较 tiling 和核数

如果已经有候选列表：

```bash
python -B -m matmul_cost_model compare --input examples/candidate_comparison.json --top 3 --output results/comparison.json
```

也可以枚举当前已审查模板：

```bash
python -B -m matmul_cost_model compare --shape 16 16 102400 --dtype fp16 --implementation deterministic --cores 4,8,16,24 --top 5 --output results/search_det.json
python -B -m matmul_cost_model compare --shape 64 128 512 --dtype bf16 --implementation all --cores 1,2,4,8 --top 3 --output results/search_bf16.json
```

只生成候选、以后再评估：

```bash
python -B -m matmul_cost_model candidates --shape 16 16 102400 --dtype fp16 --implementation deterministic --cores 4,8,16,24 --output results/candidates.json
python -B -m matmul_cost_model compare --input results/candidates.json --output results/ranking.json
```

`groups` 按同实现、同 shape、同 dtype 分组，`ranking` 从预测最快到最慢；`--top 0` 显示全部。`predictions` 始终保留全部候选与完整配置，不因 `--top` 丢失评估数据。不同实现可同时评分，但本模型的稳定排序验证是在**同实现、同输入**内进行，不能宣称跨实现排序全部验证通过。

候选生成器不是 ops-nn 自动 host tiler，也不保证找出任意新 tiling 空间的全局最优。生产搜索应枚举真实实现的合法计划，再对处于模型支持域的计划评分。

## 7. 在别的 Python 项目中复用

```python
from matmul_cost_model import MatmulCostModel, make_config, generate_candidates

model = MatmulCostModel()  # 加载固定参数一次，多次复用
config = make_config(
    implementation="ops_nn_deterministic_split_k",
    shape=[16, 16, 102400], dtype="fp16", cores=24,
    tiling={"base_m": 128, "base_n": 128, "base_k": 128, "step_k": 2},
)
prediction = model.predict(config)
print(prediction["predicted_us"])

plans = generate_candidates(
    [64, 128, 512], "bf16", implementations=("base",), cores=(1, 2, 4, 8)
)
ranking = model.compare(plans, top=3)
print(ranking["groups"][0]["ranking"])
```

直接运行现成示例：`python -B examples/use_api.py`。这个脚本使用相对于自身的路径，因此可从其他工作目录启动。

`model.predict_many(configs)` 支持批量评分。公共入口不注入顶层 `costmodel` 或 `design` 模块，减少与调用者项目的模块命名冲突。固定递推使用共享缓存，公共 API 通过锁保护；大量候选可用进程分片，但单进程复用同一实例更便于避免反复加载参数。

## 8. 查看语义流水公式 P

默认是固定校准预测 `T`；使用 `--semantic` 得到去掉外层残差后的静态流水 `P`：

```bash
python -B -m matmul_cost_model predict --input examples/fp16_deterministic.json --semantic --output results/semantic.json
python -B -m matmul_cost_model predict --input examples/bf16_base.json --explain --output results/explanation.json
```

`--explain` 保留公式中间值和语义流水分组账目，会比默认预测多做一些计算。输出中 `formula` 区分 `fixed_calibrated` 与 `semantic_pipeline`。P与T之差不是实测误差；只有对真实计时才可算误差。公式和各项解释见 [docs/MODEL_AND_SCOPE.md](docs/MODEL_AND_SCOPE.md)。

## 9. 和自己的实测结果比较

代码不包含 NPU benchmark kernel；它可以把已有 benchmark 的计时结果与预测对齐比较。先确保实测的实现、tiling、核数、dtype、输出类型及执行协议相同。

CSV 的单位必须是**微秒**：

```csv
case_id,process_index,measured_us
my_det24,0,93.2
my_det24,1,93.5
my_det24,2,93.1
```

上面的数字仅说明 CSV 格式。随包的 `examples/measured_times.csv` 则是12个候选的真实三进程结果，与 `examples/candidate_comparison.json` 一一对应，可直接运行：

```bash
python -B -m matmul_cost_model evaluate --input examples/candidate_comparison.json --measurements examples/measured_times.csv --output results/accuracy.json
```

若已保存预测，评估时可不再调用模型：

```bash
python -B -m matmul_cost_model evaluate --predictions results/comparison.json --measurements examples/measured_times.csv --output results/accuracy_cached.json
```

`case_id` 必须完整对应，缺失或重复的进程编号会报错，避免错行或单位混淆导致虚假的成绩。每个配置取不同进程值的中位数，然后计算 APE、MAPE、最大APE、5%覆盖、严格错序及稳定错序。结果中的比例用小数表示，例如 `mape=0.04` 就是4%。

稳定对需要双方至少三个进程值、真实差距大于5%、进程范围不重叠。只有一次计时时可以评估误差与严格顺序，但 `stable5` 和稳定首选统计为 `null`，不会把缺少复测说成零稳定错序。输入的每个进程值应先是该进程预热后的批量计时聚合值，不要把三次相邻单次 launch 当作三个独立进程。

评估代码只计算统计，不更新参数，也不向预测器提供实测数据。所有随包独立标签均不得用于回拟后再声称独立通过。

## 10. 自测、完整验证和 CPU 开销

快速检查，全部只用标准库：

```bash
python -I -S -B tools/verify_release.py
```

复算全部1,198个冻结独立配置，核对预测值以及原始误差/排序统计：

```bash
python -I -S -B tools/verify_release.py --full --output results/full_validation.json
```

这个命令检查交付 SHA256、接口、参数拒绝、计划往返、原始八个示例、命令行和比较功能；完整模式还禁用网络与子进程执行，并核对推理模块全部来自当前目录。`-I -S` 避免依赖调用者的 PYTHONPATH 和第三方 site-packages。1198配置复算需要一定 CPU 时间，进度输出到 stderr，最终 JSON 输出到 stdout 或指定文件。

测试模型**自身的 CPU 评分开销**：

```bash
python -B tools/benchmark_cpu.py --input examples/fp16_deterministic.json --repeat 10
```

`mean_scoring_CPU_ms` 是 Python 程序评分耗时；`predicted_kernel_us` 是预测的设备kernel耗时，两者不是一个量。这个脚本不执行 NPU kernel。

## 11. 当前支持域和精度

| 项目 | 本次支持/限制 |
|---|---|
| 硬件 | Ascend 910B2，原离线校准的软件/实现条件 |
| 语义 | NN、batch=1、FP16/BF16、最终对应16位输出 |
| M | 16到128，16的倍数 |
| N | 16、24、32、64、96、128 |
| K | 64到131072，16的倍数 |
| 启动核数 | 1到24；DET至少2核且K分片非空 |
| 布局/地址 | 已审查的紧凑输入和转换规则，无额外stride/offset；不支持任意改变缓存策略 |
| 执行协议 | 已冻结的预热协议；预测程序自身不会执行预热或计时 |

已经审查的 tiling：

| 实现 | baseM×baseN×baseK / stepK，或 L1/L0 |
|---|---|
| CAT default | L1=128×256×256，L0=128×256×64 |
| CAT deepk | L1=16×16×4096，L0=16×16×1024 |
| CAT widek | L1=16×32×2048，L0=16×32×512 |
| CAT midk | L1=32×64×1024，L0=32×64×256 |
| BASE | 16×16×128/s4；32×64×128/s1,s4,s8；64×32×128/s4；128×128×128/s4 |
| 核内 Split-K | 16×16×128/s4；32×64×128/s2,s4,s8；64×32×128/s4；128×128×128/s4 |
| 确定性 Split-K | 128×128×128/s1,s2,s3 |

最新独立验证是8个未见MNK、每dtype599个配置、四实现、三个正式进程：

| dtype | MAPE | 最大APE | 5%覆盖 | 稳定错序 |
|---|---:|---:|---:|---:|
| FP16 | 4.00278% | 16.56643% | 66.77796% | 30/5028 |
| BF16 | 4.53790% | 18.79587% | 64.77462% | 27/4984 |

四种实现各自的平均误差均低于5%。各dtype的32个同输入/同实现分组中，预测首选均未被其他候选稳定击败，最大原始首选损失为1.18%/1.53%。**全候选稳定排序仍有错误，不能承诺未知输入零错序，也不能把平均5%理解为每个点都在5%内。** 每种实现的详细成绩在 `data/model_info.json`。

FP32、转置、batch>1、任意新tiling/缓冲/地址、冷缓存、其他硬件/SDK，以及完整融合Attention都未获得本次资格。即便输入落在合法接口区间，也不意味着每个未见组合自动具备5%/20%的数学保证。融合Attention中的驻留、共享、启动与转换不能用独立GEMM耗时机械相加。

本次交付的数值复算、隔离运行、使用命令和安装检查记录见 `RELEASE_VERIFICATION.json`。它记录的是封装一致性与已有独立成绩的复现，不是一次新的硬件精度实验。
