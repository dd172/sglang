# RadixAttention 单卡复现

在同一块 GPU 和同一模型上运行两次 SGLang。两次启动之间只切换
`--disable-radix-cache`。每轮顺序提交五条 2,176 token 的请求，每条只生成
1 个 token。直接提交 token ID，以便精确控制共享前缀的长度。

上游基线是 SGLang `v0.5.6`（`7ae368efde14f0b2abe769a5fbb99b8154b2a9b7`）。
默认模型是 Qwen2.5-0.5B-Instruct，固定到 Hugging Face 提交
`7ae557604adf67be50417f59c2c2f167def9a775`。使用单张 A40、FP16、Triton
attention、1 token 的 KV cache 页大小和先到先服务调度。

## 安装并运行

需要 Linux、支持 CUDA 12.8 的 NVIDIA 驱动、Python 3.10 或 3.11，以及
`venv`。工作目录、临时文件和下载缓存都可以放在有充足空间的用户目录中。

```bash
mkdir -p "$HOME/radix-repro"
cd "$HOME/radix-repro"
git clone --depth 1 --branch radix-attention-repro https://github.com/dd172/sglang.git
cd sglang

mkdir -p "$PWD/.scratch" "$PWD/.pip-cache"
export TMPDIR="$PWD/.scratch" PIP_CACHE_DIR="$PWD/.pip-cache"
export HF_HOME="$HOME/radix-repro/huggingface"
export TRITON_CACHE_DIR="$HOME/radix-repro/triton"

python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -e ./python
.venv/bin/python -m pip check

.venv/bin/python experiments/radix-attention/run.py \
  --rounds 10 --output "$HOME/radix-repro/results/$(date -u +%Y%m%dT%H%M%SZ)"
```

脚本默认选第一张已用显存低于 512 MiB 的 A40，也可以用 `--gpu` 指定 GPU
的编号或 UUID。首次启动会下载模型。已有相同版本的模型文件时，可以用
`--model-path /path/to/local/snapshot` 指定本地快照。

## 验收标准

| 请求 | 开启时预期命中 token | 关闭时预期命中 token |
| --- | ---: | ---: |
| 首次填充缓存 | 0 | 0 |
| 共享前 2,048 个 token，随后分叉 | 2,048 | 0 |
| 共享前 1,024 个 token，随后分叉 | 1,024 | 0 |
| 首 token 不同，剩余 token 与首请求完全相同 | 0 | 0 |
| 原样重放首请求 | 2,175 | 0 |

计数来自 SGLang 原生接口的 `meta_info.cached_tokens`。原样重放时，
SGLang v0.5.6 会重新处理最后一个 prompt token 来产生首个输出 token，
因此预期命中为 2,175。脚本逐条核对命中数和输入长度，偏离预期就报错。

每轮先清空缓存。每种配置先跑两轮预热，随后统计各类请求的
首 token 延迟（TTFT）中位数。TTFT 从本地客户端发起请求到收到含首 token
的 SSE 事件，包含本地接口开销。生成 token 的一致次数也会记录。该实验覆盖
论文中的前缀复用机制。复现论文的应用吞吐量需要原工作负载和对应的对照系统。

输出目录下有 `manifest.json`、逐请求 `requests.jsonl`、`summary.json`
和两份服务器日志。预热轮编号为 -2 和 -1，仅保留在逐请求数据中。

参考 [论文 §3 RadixAttention](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf)、
[SGLang 上游实现](https://github.com/sgl-project/sglang/tree/v0.5.6/python/sglang/srt/mem_cache)
和 [PyTorch 安装版本](https://pytorch.org/get-started/previous-versions/#v291)。
