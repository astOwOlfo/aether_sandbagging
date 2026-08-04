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


async def run_experiments() -> None:
    models: list[Model] = [
        Model("anthropic/claude-sonnet-4.6"),
        Model("deepseek/deepseek-v4-pro"),
        Model("google/gemma-4-31b-it"),
        Model("openai/gpt-oss-120b"),
        Model("qwen/qwen3.6-35b-a3b"),
    ]

    dataset = load_literature_review_data()

    extractor = Model("z-ai/glm-5.2")
    judge = Model("deepseek/deepseek-v4-flash-0731", web_search=True, max_parallel=256)
    refusal_judge = Model("z-ai/glm-5.2")

    results: list[ExperimentResult] = await asyncio.gather(
        *[
            run_experiment(
                dataset=dataset,
                resamples=4,
                model=model,
                extractor=extractor,
                judge=judge,
                refusal_judge=refusal_judge,
                strict_refusal_judge=False,
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

    title: str = (
        "Literature Reviews Generated with Malicious and Control Prompt Paraphrases"
    )
    plot_dir: str = "plots/literature_review/"
    plot_filename: str = os.path.join(plot_dir, "plot.html")
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
        papers_only=False,
        model_legend=False,
    )

    visualization_dir: str = "visualizations/literature_review/"
    visualization_filename: str = os.path.join(visualization_dir, "visualization.html")
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
    await asyncio.gather(run_experiments())


if __name__ == "__main__":
    asyncio.run(main())


# $142.524
