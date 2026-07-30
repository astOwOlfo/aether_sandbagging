import os
import asyncio

from src.healthbench import (
    run_experiment,
    ExperimentResult,
    load_healthbench_data,
    Datapoint,
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
                seed=42,
                tqdm_description=model.model.split("/")[-1],
            )
            for model in models
        ]
    )

    plot_model_name_map: dict[str, str] = {
        "huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2": "gemma-4-31B-it-abliterated-v2",
        "llmfan46/gpt-oss-120b-heretic-v2": "gpt-oss-120b-heretic-v2",
        "DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark": "Qwen3.6-35B-A3B-Uncensored-Aggressive",
    }

    filename_suffix: str = ""
    title: str = "HealthBench Scores with Malicious and Control Prompt Paraphrases"
    if helpful_only_models:
        filename_suffix += "_helpful_only"
        title += "<br>Helpful-Only Models"
    if strict_refusal_judge:
        filename_suffix += "_strict_exclusion"
        title += "<br>Strict Exclusion"

    plot_dir: str = "plots/healthbench/"
    plot_filename: str = os.path.join(plot_dir, f"plot{filename_suffix}.html")
    os.makedirs(plot_dir, exist_ok=True)
    plot(
        models=models,
        results=results,
        title=title,
        healthbench_grader=grader.model.split("/")[-1],
        refusal_judge=judge.model.split("/")[-1],
        statement_correctness_judge=judge.model.split("/")[-1],
        html_filename=plot_filename,
        model_name_map=plot_model_name_map,
    )

    visualization_dir: str = "visualizations/healthbench/"
    visualization_filename: str = os.path.join(
        visualization_dir, f"visualization{filename_suffix}.html"
    )
    os.makedirs(visualization_dir, exist_ok=True)
    visualize(
        models=models,
        results=results,
        title=title,
        healthbench_grader=grader.model.split("/")[-1],
        refusal_judge=judge.model.split("/")[-1],
        statement_correctness_judge=judge.model.split("/")[-1],
        html_filename=visualization_filename,
        model_name_map=plot_model_name_map,
    )


async def main() -> None:
    await asyncio.gather(
        run_experiments(strict_refusal_judge=False, helpful_only_models=False),
        run_experiments(strict_refusal_judge=True, helpful_only_models=False),
        # run_experiments(strict_refusal_judge=False, helpful_only_models=True),
        # run_experiments(strict_refusal_judge=True, helpful_only_models=True),
    )


if __name__ == "__main__":
    asyncio.run(main())
