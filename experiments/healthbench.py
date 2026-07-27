from dotenv import load_dotenv
import asyncio

from src.healthbench import (
    run_experiment,
    ExperimentResult,
    load_healthbench_data,
    Datapoint,
    print_experiment_result,
)
from src.llm_apis import Model
from src.visualize_healthbench import visualize


async def main(prefilter_malicious_refusals: bool) -> None:
    dataset: list[Datapoint] = load_healthbench_data()
    dataset = dataset

    models: list[Model] = [
        # Model("huihui-ai/Huihui-gemma-4-31B-it-abliterated-v2", provider="vllm"),
        # Model("llmfan46/gpt-oss-120b-heretic-v2", provider="vllm"),
        # Model(
        #     "DreamFast/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Safetensor-Benchmark",
        #     provider="vllm",
        # ),
        # Model("anthropic/claude-sonnet-4.6"),
        # Model("deepseek/deepseek-v4-pro"),
        Model("google/gemma-4-31b-it"),
        # Model("openai/gpt-oss-120b"),
        # Model("qwen/qwen3.6-35b-a3b"),
    ]

    resamples: int = 32 if prefilter_malicious_refusals else 4
    datapoints: int | None = 32 if prefilter_malicious_refusals else None
    grader: Model = Model("deepseek/deepseek-v4-flash", max_parallel=1024)
    judge = Model("z-ai/glm-5.2")

    args: list[tuple[Model, bool]] = [
        (model, strict_refusal_judge)
        for model in models
        for strict_refusal_judge in [False, True]
    ]

    results: list[ExperimentResult] = await asyncio.gather(
        *[
            run_experiment(
                dataset=dataset,
                model=model,
                resamples=resamples,
                datapoints=datapoints,
                grader=grader,
                refusal_judge=judge,
                strict_refusal_judge=strict_refusal_judge,
                claim_extractor=judge,
                claim_correctness_judge=judge,
                prefilter_malicious_refusals=prefilter_malicious_refusals,
                seed=42,
            )
            for model, strict_refusal_judge in args
        ]
    )

    for result, (model, strict_refusal_judge) in zip(results, args, strict=True):
        print("=" * 64)
        print("model:", model)
        print("strict refusal judge:", strict_refusal_judge)
        print_experiment_result(result)

        visualize(
            scores=result.scores,
            html_filename=f"visualizations/healthbench-sandbagging-{model.model.split('/')[-1]}-{resamples}{'-prefilter-malicious-refusals' if prefilter_malicious_refusals else ''}-resamples{'-strict-refusal-judge' if strict_refusal_judge else ''}.html",
        )


if __name__ == "__main__":
    asyncio.run(main(prefilter_malicious_refusals=False))
