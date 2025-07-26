# D2F
Discrete Diffusion Forcing

## vLLM Implementation

Based on [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm).

## Configure the Project

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