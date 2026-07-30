import os
import asyncio

from src.literature_review import (
    ExperimentResult,
    load_literature_review_data,
    run_experiment,
)
from src.plot_literature_review import plot
from src.llm_apis import Model


async def run_experiments(
    strict_refusal_judge: bool, helpful_only_models: bool
) -> None:
    if helpful_only_models:
        models: list[Model] = [
            Model(
                "huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2",
                provider="vllm",
                max_parallel=256,
            ),
            Model(
                "llmfan46/gpt-oss-120b-heretic-v2",
                provider="vllm",
                max_parallel=256,
            ),
            Model(
                "DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
                provider="vllm",
                max_parallel=256,
            ),
        ]
    else:
        models: list[Model] = [
            Model("anthropic/claude-sonnet-4.6"),
            # Model("deepseek/deepseek-v4-pro"),
            # Model("google/gemma-4-31b-it"),
            # Model("openai/gpt-oss-120b"),
            # Model("qwen/qwen3.6-35b-a3b"),
        ]

    dataset = load_literature_review_data()[:32]

    extractor = Model("z-ai/glm-5.2")
    judge = Model("z-ai/glm-5.2", web_search=True)
    refusal_judge = Model("z-ai/glm-5.2")

    results: list[ExperimentResult] = await asyncio.gather(
        *[
            run_experiment(
                dataset=dataset,
                resamples=1,
                model=model,
                extractor=extractor,
                judge=judge,
                refusal_judge=refusal_judge,
                strict_refusal_judge=strict_refusal_judge,
                seed=42,
                tqdm_description=model.model.split("/")[-1],
            )
            for model in models
        ]
    )

    plot_dir: str = "plots/literature_review/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_file: str = "plot"
    if strict_refusal_judge:
        plot_file += "_strict_refusal_judge"
    if helpful_only_models:
        plot_file += "_helpful_only_models"
    plot_file += ".html"
    plot(
        models=models, results=results, html_filename=os.path.join(plot_dir, plot_file)
    )


async def main() -> None:
    await asyncio.gather(
        run_experiments(strict_refusal_judge=False, helpful_only_models=False),
        # run_experiments(strict_refusal_judge=True, helpful_only_models=False),
        # run_experiments(strict_refusal_judge=False, helpful_only_models=True),
        # run_experiments(strict_refusal_judge=True, helpful_only_models=True),
    )


if __name__ == "__main__":
    asyncio.run(main())
