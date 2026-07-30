# Model Organisms of Sandbagging in the Wild

This is the code for the [Model Organisms of Sandbagging in the Wild blogpost](https://docs.google.com/document/d/1SmZZKFO6GVbIFWQ4lGPh92cpGUgHvijLXOsEdXVCGYM/edit?tab=t.nqr3jktbrlk6).

## Cloning

This repo uses [Git LFS](https://git-lfs.com) for large files.
Install it before cloning:

```bash
git lfs install
```

Then clone normally:

```bash
git clone https://github.com/you/your-repo.git
```

If you already cloned without LFS, run:

```bash
git lfs pull
```

## Running

Create a `.env` file with an `OPENROUTER_API_KEY=...` line. Then, run:

```bash
uv run python -m experiments.healthbench
uv run python -m experiments.literature_review
```

This generates plots in `plots/` and transcript visualizations in `visualizations/`.

To run experiments with helpful-only models: Uncomment the calls to `run_experiments` with `helpful_only=True` in `experiments/healthbench.py` and `experiments/literature_review.py`. Comment all uncensored model names but one. Install [vllm](https://pypi.org/project/vllm/) on a machine with GPUs and serve an OpenAI compatible API for the model using `vllm serve` on port 8000. Run the experiment files. Repeat for all helpful-only models. Completions are cached, so once you ran experiments with some model, you don't need to serve it to rerun the experiments. Use the following commands to serve the models:

```bash
git clone https://github.com/vllm-project/vllm # only needed for the gemma chat template
vllm serve huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2 --reasoning-parser gemma4 --chat-template vllm/examples/tool_chat_template_gemma4.jinja

vllm serve llmfan46/gpt-oss-120b-heretic-v2

vllm serve DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark --reasoning-parser qwen3
```
