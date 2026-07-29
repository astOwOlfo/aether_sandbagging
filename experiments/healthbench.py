import os
import asyncio

from src.healthbench import (
    filter_malicious_compliances,
    run_experiment,
    ExperimentResult,
    load_healthbench_data,
    Datapoint,
    print_experiment_result,
)
from src.llm_apis import Model
from src.plot_healthbench import plot
from src.visualize_healthbench import visualize


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
            Model("deepseek/deepseek-v4-pro"),
            Model("google/gemma-4-31b-it"),
            Model("openai/gpt-oss-120b"),
            Model("qwen/qwen3.6-35b-a3b"),
        ]

    dataset: list[Datapoint] = load_healthbench_data()

    grader = Model("deepseek/deepseek-v4-flash", max_parallel=256)
    judge = Model("z-ai/glm-5.2")

    results: list[ExperimentResult] = await asyncio.gather(
        *[
            run_experiment(
                dataset=dataset,
                model=model,
                resamples=4,
                datapoints=None,
                grader=grader,
                refusal_judge=judge,
                strict_refusal_judge=strict_refusal_judge,
                claim_extractor=judge,
                claim_correctness_judge=judge,
                prefilter_malicious_compliances=False,
                seed=42,
                tqdm_description=model.model.split("/")[-1],
            )
            for model in models
        ]
    )

    plot_dir: str = "plots/healthbench/"
    os.makedirs(plot_dir, exist_ok=True)
    plot_file: str = "plot"
    if strict_refusal_judge:
        plot_file += "_strict_refusal_judge"
    if helpful_only_models:
        plot_file += "_helpful_only_models"
    plot_file += ".html"
    plot(
        models=models,
        results=results,
        html_filename=os.path.join(plot_dir, plot_file),
    )

    """
    for model, result in zip(models, results, strict=True):
        print("=" * 64)
        print("model:", model)
        print("strict refusal judge:", strict_refusal_judge)
        print_experiment_result(result)

        visualize(
            scores=result.scores,
            html_filename=f"visualizations/healthbench-sandbagging-{model.model.split('/')[-1]}-{resamples}{'-prefilter-malicious-refusals' if prefilter_malicious_refusals else ''}-resamples{'-strict-refusal-judge' if strict_refusal_judge else ''}.html",
        )
    """


async def main() -> None:
    await asyncio.gather(
        run_experiments(strict_refusal_judge=False, helpful_only_models=False),
        # run_experiments(strict_refusal_judge=True, helpful_only_models=False),
        # run_experiments(strict_refusal_judge=False, helpful_only_models=True),
        # run_experiments(strict_refusal_judge=True, helpful_only_models=True),
    )


if __name__ == "__main__":
    asyncio.run(main())
