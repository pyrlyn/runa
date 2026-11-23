# Hardware survey and `load --auto`

`runa runtime survey` shows what a machine has to run models with. `runa load --auto` lets the
daemon choose the GPU offload and the context length of a model by the same survey.

## `runa runtime survey`

```console
$ runa runtime survey
CPU: Apple M3 Pro (arm64), 12 cores (6 performance), 12 threads
Memory: 36.0 GB
GPU: Apple M3 Pro, 27.0 GB of the machine's memory (metal)
Engine: llama.cpp 1.0.0 (cpu, mtl, blas)
```

It is the machine the command runs on, and no daemon is asked or started. The last line is the
engine in this binary and the backends it was built with. When GPUs were found and the engine
cannot compute on any of them, one more line says so: `This build of runa computes on the CPU
only`.

Where the numbers come from:

| | macOS | Linux | Windows |
|---|---|---|---|
| CPU model | `sysctl machdep.cpu.brand_string` | `model name` in `/proc/cpuinfo` | not told |
| Cores | `hw.physicalcpu`, `hw.perflevel0.physicalcpu` | the distinct `thread_siblings_list` under `/sys/devices/system/cpu` | not told |
| Threads | `hw.logicalcpu` | the processors that are online | the active processors |
| Memory | the physical memory | the physical memory | the physical memory |
| GPUs | Metal: every device, with `recommendedMaxWorkingSetSize` as its memory | `nvidia-smi`, then `/sys/class/drm` for NVIDIA, AMD and Intel cards | not told |

On Apple Silicon the GPU has no memory of its own: its number is the share of the machine's
memory the system lets it use, and the survey says `of the machine's memory`. On Linux an NVIDIA
card is asked through `nvidia-smi` when the program is there; sysfs tells the memory of an AMD
card (`mem_info_vram_total`) and only the presence of the others.

### JSON

`--json` prints one object. A `null` is something the system did not tell.

```json
{
  "os": "macos",
  "cpu": {"model": "Apple M3 Pro", "architecture": "arm64", "cores": 12, "performance_cores": 6, "threads": 12},
  "memory_bytes": 36000000000,
  "gpus": [{"name": "Apple M3 Pro", "memory_bytes": 27000000000, "unified_memory": true, "source": "metal"}],
  "engine": {"name": "llama.cpp", "version": "1.0.0", "accelerators": ["cpu", "mtl", "blas"], "gpu_offload": true}
}
```

| Field | Content |
|---|---|
| `os` | `macos`, `linux` or `windows` |
| `cpu.cores` | Physical cores |
| `cpu.performance_cores` | The fast cores of a CPU that also has slow ones, else `null` |
| `cpu.threads` | The processors the system schedules on |
| `gpus[].memory_bytes` | The card's own memory, or with `unified_memory` the share of the machine's memory |
| `gpus[].source` | Who told about it: `metal`, `nvidia-smi` or `sysfs` |
| `engine.accelerators` | The backends the engine was built with |
| `engine.gpu_offload` | Whether the engine can put a model, or a part of one, on a GPU of this machine |

`runa status --json` has the same `engine` object for the engine of the daemon.

## `runa load --auto`

```console
$ runa load bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M --auto --estimate-only
Estimated memory: 800 MB (weights 398 MB, context 403 MB) with context 32768, gpu max
$ runa load bartowski/Qwen2.5-0.5B-Instruct-GGUF@Q4_K_M --auto
Loaded bartowski/Qwen2.5-0.5B-Instruct-GGUF@q4_k_m in 0.21 s: context 32768, gpu max, about 800 MB (weights 398 MB, context 403 MB)
bartowski/Qwen2.5-0.5B-Instruct-GGUF@q4_k_m
```

Without `--auto` a model is loaded with `--gpu max` and a context of 4096 tokens, or with what
`config.json` says, whether or not that fits. With `--auto` the daemon surveys its own machine
and takes the first of these that fits the memory that is free for models:

1. The whole model on the GPU, with the longest context that fits.
2. On a GPU with memory of its own: a part of the model on the GPU and the rest in the machine's
   memory, with a context of at most 4096.
3. The whole model on the CPU, with the longest context that fits.

The contexts that are tried are the one the model was trained with, at most 32768, and then
32768, 16384 and so on down to 2048. When nothing fits, the load is refused and the answer says
how much the model takes and how much there is; `--gpu` and `--context-length` without `--auto`
load it all the same.

What is given is kept: `--auto --gpu off` chooses the context only, `--auto -c 8192` the GPU
offload only. The defaults of `config.json` step aside for `--auto`.

How the memory is counted:

- Of the machine's memory a fifth, and at least 2 GiB, is left to the system and to the buffers
  the estimate does not count.
- Of a GPU's own memory 90 % is used. On Apple Silicon the GPU's share is a part of the
  machine's memory, so a model has to fit both numbers.
- What the models that are loaded already were estimated at is subtracted. A load that is still
  in progress is not counted.
- With several GPUs the one with the most memory is planned for.
- A GPU the engine cannot compute on is not counted at all.

`--auto --estimate-only` shows what would be chosen and loads nothing. In `--json` the choice is
the `auto` object, which is `null` without `--auto`:

```json
{"status": "estimated", "estimate": {"...": "..."}, "auto": {"context_length": 32768, "gpu": "max"}}
```

The daemon decides, not the command: with `--host` it is the machine of the daemon that is
surveyed. `--auto` is refused with `--backend lmstudio`, which has its own way of choosing.
