import os
import asyncio

from src.literature_review import (
    ExperimentResult,
    load_literature_review_data,
    run_experiment,
)
from src.llm_apis import Model
from src.plot_literature_review import plot
from src.visualize_literature_review import visualize


async def run_experiments(
    strict_refusal_judge: bool,
    helpful_only_models: bool,
    simplified_plot_only: bool = False,
) -> None:
    if helpful_only_models:
        models: list[Model] = [
            Model(
                "huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2",
                provider="vllm",
                max_parallel=256,
            ),
            Model(
                "DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
                provider="vllm",
                max_parallel=256,
            ),
            Model(
                "llmfan46/gpt-oss-120b-heretic-v2",
                provider="vllm",
                max_parallel=256,
            ),
        ]
    else:
        models: list[Model] = [
            Model("google/gemma-4-31b-it"),
            Model("qwen/qwen3.6-35b-a3b"),
            Model("meta-llama/llama-3.3-70b-instruct"),
            Model("z-ai/glm-5.2"),
            Model("mistralai/mistral-small-2603"),
        ]

    dataset = load_literature_review_data()

    extractor = Model("z-ai/glm-5.2")
    judge = Model("z-ai/glm-5.2", web_search=True)
    refusal_judge = Model("z-ai/glm-5.2")

    results: list[ExperimentResult] = await asyncio.gather(
        *[
            run_experiment(
                dataset=dataset,
                resamples=1 if helpful_only_models else 4,
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

    plot_model_name_map: dict[str, str] = {
        "huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2": "gemma-4-31B-it-abliterated",
        "DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark": "qwen3.6-35b-a3b-Uncensored",
        "llmfan46/gpt-oss-120b-heretic-v2": "gpt-oss-120b-heretic",
    }

    filename_suffix: str = ""
    title: str = (
        "Literature Reviews Generated with Malicious and Control Prompt Paraphrases"
    )
    if simplified_plot_only:
        filename_suffix += "_simplified"
    if helpful_only_models:
        filename_suffix += "_helpful_only"
        title += "<br>Helpful-Only Models"
    if strict_refusal_judge:
        filename_suffix += "_strict_exclusion"
        title += "<br>Strict Exclusion"

    plot_dir: str = "plots/literature_review/"
    plot_filename: str = os.path.join(plot_dir, f"plot{filename_suffix}.html")
    os.makedirs(plot_dir, exist_ok=True)
    plot(
        models=models,
        results=results,
        title=title,
        claim_extractor=extractor.model.split("/")[-1],
        correctness_judge=judge.model.split("/")[-1]
        + (" with internet access" if judge.web_search else ""),
        refusal_judge=refusal_judge.model.split("/")[-1],
        html_filename=plot_filename,
        model_name_map=plot_model_name_map,
        papers_only=simplified_plot_only,
        model_legend=False,
    )

    if simplified_plot_only:
        return

    visualization_dir: str = "visualizations/literature_review/"
    visualization_filename: str = os.path.join(
        visualization_dir, f"visualization{filename_suffix}.html"
    )
    os.makedirs(visualization_dir, exist_ok=True)
    visualize(
        models=models,
        results=results,
        title=title,
        claim_extractor=extractor.model.split("/")[-1],
        correctness_judge=judge.model.split("/")[-1]
        + (" with internet access" if judge.web_search else ""),
        refusal_judge=refusal_judge.model.split("/")[-1],
        html_filename=visualization_filename,
        model_name_map=plot_model_name_map,
        dataset=dataset,
    )


async def main() -> None:
    await asyncio.gather(
        run_experiments(
            strict_refusal_judge=False,
            helpful_only_models=False,
            simplified_plot_only=True,
        ),
        run_experiments(strict_refusal_judge=False, helpful_only_models=False),
        run_experiments(strict_refusal_judge=True, helpful_only_models=False),
        run_experiments(strict_refusal_judge=False, helpful_only_models=True),
        run_experiments(strict_refusal_judge=True, helpful_only_models=True),
    )


if __name__ == "__main__":
    asyncio.run(main())


# $371.498
