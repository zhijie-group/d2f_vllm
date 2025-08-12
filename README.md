# D2F-vLLM

vLLM implementation for Diffusion LLMs, D2F is integrated as the core inference strategy, while also support training-free strategies like Fast-dLLM.

## Foundation of Our vLLM Implementation

Based on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

## Easy Install D2F-vLLM

```shell
pip install d2f_vllm
```

## Configure the Project from Source (for Developers)

We use [UV](https://github.com/astral-sh/uv) to manage the whole project. 

### Install UV

[UV Installation](https://docs.astral.sh/uv/getting-started/installation/)

### Initialize the Project

```shell
uv sync
source .venv/bin/activate
uv pip install -e .
```

For easy-activation:

```shell
echo "alias uvon=source .venv/bin/activate" >> ~/.zshrc # If using bash, change to .bashrc
source ~/.zshrc
```

Then, use `uvon` under the project root path to activate.

### Download Flash Attention

```shell
uv pip install flash-attn --no-build-isolation
```

If not working, build `flash-attn` from scratch. This may take some while (most of the time is cost on compiling `cutlass`).

```shell
git submodule update --init --recursive
cd third_party/flash-attn
MAX_JOBS=$(nproc) python setup.py install --verbose
```

## User Guideline

### Setting Generation Mode

Setting `add_new_block_threshold=1.0` allows compatibility with all diffusion LLM decoding paradigms.

In contrast, setting `add_new_block_threshold<1.0`, together with our `D2F` training strategy, enables support for the D2F-specific decoding paradigm.

## TODO List

- [ ] Implement KV Cache loading kernel
- [ ] Implement Async Engine and Streaming Generation
- [ ] Faster Flash Attention Kernel
- [ ] Diffusion LM CUDA Graph Capturing
- [ ] ...