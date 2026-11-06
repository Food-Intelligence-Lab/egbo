#!/usr/bin/env python3
"""Query the formulation expert once for a global BO configuration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from egbo_bo_config import BOConfig, _extract_json
from run_formulatebench import OpenRouterClient


SYSTEM_PROMPT = (
    "You are an expert in Bayesian optimization configuring a "
    "Gaussian-process optimizer for an expensive scientific experiment."
)

USER_PROMPT = """We are optimizing the overall similarity of plant-based formulations to target animal products. An expert first proposes a recipe containing at most 9 ingredients and specific mixture fractions. Bayesian optimization then refines the mixture fractions over the simplex.

The evaluator is a black box. After each experiment, the optimizer receives one scalar target-dissimilarity score, where lower is better. The total budget is 20 evaluations, including the expert's initial recipe. No information is available about the mathematical form, smoothness, or noise level of the hidden objective.

Choose one configuration that will be fixed across all product categories. Select exactly one value for each field:

- kernel: "matern52", "matern32", or "rbf"
- lengthscale_prior: "default", "long", or "short"
- noise: "default", "low", or "fixed_tiny"
- acquisition: "qlognei", "qlogei", or "qucb"
- ard: "ard" or "isotropic"

Return only a JSON object in this form:
{
  "kernel": "matern52",
  "lengthscale_prior": "default",
  "noise": "default",
  "acquisition": "qlognei",
  "ard": "isotropic"
}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic/claude-opus-4.7")
    parser.add_argument("--cache-dir", type=Path, default=Path("bo_config_matched_cache_v1"))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/bo_config_matched_v1.json"),
    )
    args = parser.parse_args()

    client = OpenRouterClient(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=args.model,
        cache_dir=str(args.cache_dir),
    )
    raw = client.chat(SYSTEM_PROMPT, USER_PROMPT, temperature=0.0)
    requested = _extract_json(raw)
    config = BOConfig.from_spec(requested, context="matched-global-query")
    payload = {
        "model": args.model,
        "temperature": 0.0,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "raw_response": raw,
        "requested_config": requested,
        "validated_config": {
            "kernel": config.kernel,
            "lengthscale_prior": config.lengthscale_prior,
            "noise": config.noise,
            "acquisition": config.acquisition,
            "ard": config.ard,
        },
        "config_label": config.label(),
        "validation_fallbacks": config._fallbacks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["validated_config"], indent=2))
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
