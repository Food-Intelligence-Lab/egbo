#!/usr/bin/env python3
"""Tool-using LLM agent on FormulateBench: does the agent hand off to an optimizer?

The agent receives the SAME task prompt as the semantic-only iterative LLM
baseline (`iterative_llm_optimizer`, unrestricted arm): the target category, the
ingredient catalog (names only), and the evaluated history starting from the
model's own shared first recipe. Only the final instruction paragraph differs,
because the agent acts through tools instead of returning one JSON recipe:

  evaluate(recipe)  submits one formulation (<= 9 catalog ingredients) to the
                    black-box evaluator; 20 evaluations in total, including the
                    shared first recipe.
  python(code)      runs Python in a sandbox (numpy, scipy, scikit-learn and
                    pandas named; no network; no access to the benchmark code,
                    data or user files). evaluations.json in the working
                    directory always holds the evaluated history.

Nothing in the prompt mentions Bayesian optimization, surrogate models, or
noise. Observation noise: y_obs = y + N(0, (level * norm)^2), as in
run_noise_sweep.py. Scoring matches the other noise experiments: the true loss
of the recommended recipe (best observed so far); the agent's own final pick is
also recorded. Full transcripts (including all code the agent ran) are saved.

Results: results/agent_pilot_v1/<model_slug>/<domain>/<category>/agent_noise<level>_seed<k>.json
         (+ .transcript.json). Requires OPENROUTER_API_KEY. macOS (sandbox-exec).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = HERE / "results"
BUDGET, MAX_ING, MAX_TURNS, MAX_NUDGES = 20, 9, 120, 5
PY_TIMEOUT, OUT_CHARS = 180, 8000
MODELS = {  # model id -> first-recipe root (None = the paper's cached Opus 4.7 recipe)
    "anthropic/claude-opus-4.7": None,
    "openai/gpt-6.1-sol": str(R / "multi_llm_v1" / "openai__gpt-6.1-sol"),
    "google/gemini-4-argon": str(R / "multi_llm_v1" / "google__gemini-4-argon"),
}

TOOLS = [
    {"type": "function", "function": {
        "name": "evaluate",
        "description": "Submit one formulation to the black-box evaluator. Returns its target dissimilarity "
                       "(lower is better). Each call uses one evaluation from the budget.",
        "parameters": {"type": "object", "properties": {"recipe": {
            "type": "array", "description": "Ingredients and fractions; fractions are normalized to sum to 1.",
            "items": {"type": "object", "properties": {
                "ingredient": {"type": "string", "description": "exact catalog name"},
                "fraction": {"type": "number"}}, "required": ["ingredient", "fraction"]}}},
            "required": ["recipe"]}}},
    {"type": "function", "function": {
        "name": "python",
        "description": "Run Python code in a sandbox and return its stdout and stderr.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]


HINT = ("Measurements are noisy and each evaluation is expensive, so consider carefully how best to "
        "choose each experiment.")


def task_prompt(fb, df, category, history, hint=False, name_botorch=False):
    """Byte-identical to the semantic-only baseline prompt except the last paragraph."""
    scalar_history = [{"iteration": h["iteration"], "recipe": h["recipe"], "target_dissimilarity": h["loss"]}
                      for h in history]
    feedback = ("Previous evaluated formulations follow. "
                "Target dissimilarity is the evaluator's scalar output; "
                "lower is better. Use only these observations to propose "
                "the next formulation.\n" + json.dumps(scalar_history, separators=(",", ":")))
    remaining = BUDGET - len(history)
    libs = "numpy, scipy, scikit-learn, pandas, and BoTorch" if name_botorch else "numpy, scipy, scikit-learn and pandas"
    return f"""You are optimizing the overall similarity of a plant-based formulation to animal-based {category}. The evaluator is a black box.

Available plant-based ingredients:
{fb._ingredient_names_block(df)}

{feedback}

You have {BUDGET} evaluations in total, of which {remaining} remain. Submit formulations to the evaluator with the evaluate tool, one formulation per call. Each formulation uses at most {MAX_ING} available ingredients, nonnegative quantities, and no ingredient outside the catalog; fractions are normalized to sum to 1. You can also run Python code with the python tool ({libs} are installed; there is no internet access). The working directory contains evaluations.json, which always holds all evaluations so far in the format above; each python call runs in a fresh process, and files you write in the working directory persist. Continue until you have used all evaluations.""" + (" " + HINT if hint else "")


# ---------------------------------------------------------------- sandbox
def _profile(workdir: Path) -> str:
    venv, base = Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()
    deny = {Path.home().resolve(), HERE.parent.resolve(), Path(tempfile.gettempdir()).resolve(),
            Path("/private/tmp"), Path("/private/var/folders")}
    rules = ["(version 1)", "(allow default)", "(deny network*)"]
    rules += [f'(deny file-read* (subpath "{d}"))' for d in sorted(map(str, deny))]
    rules += [f'(allow file-read* (subpath "{p}"))' for p in (base, venv, workdir)]
    rules += ["(deny file-write*)", f'(allow file-write* (subpath "{workdir}"))',
              '(allow file-write* (literal "/dev/null"))',
              '(allow file-read* file-write* (regex #"^/private/tmp/__KMP_REGISTERED_LIB_"))']
    return "\n".join(rules)


def run_python(code: str, workdir: Path, n: int) -> str:
    cell = workdir / f"_cell{n:03d}.py"
    cell.write_text(code)
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "TMPDIR": str(workdir), "MPLCONFIGDIR": str(workdir),
           "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MPLBACKEND": "Agg"}
    try:
        p = subprocess.run(["/usr/bin/sandbox-exec", "-p", _profile(workdir), sys.executable, "-s", cell.name],
                           cwd=workdir, env=env, capture_output=True, text=True, timeout=PY_TIMEOUT)
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
        if p.returncode:
            out += f"\n[exit code {p.returncode}]"
    except subprocess.TimeoutExpired:
        out = f"[timed out after {PY_TIMEOUT} s]"
    out = out.replace(str(workdir), ".").replace(str(Path(sys.prefix)), "<python-env>")
    return out if len(out) <= OUT_CHARS else out[:OUT_CHARS] + f"\n[output truncated at {OUT_CHARS} characters]"


# ---------------------------------------------------------------- API
def chat(fb, model, messages, tools, tool_choice="auto", temperature=0.0, key_env="OPENROUTER_API_KEY"):
    body = json.dumps({"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice,
                       "temperature": temperature, "usage": {"include": True}}).encode()
    last = None
    for attempt in range(8):
        fb._throttle(f"{model}@{key_env}")  # rate limits are per model and per account
        req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {os.environ[key_env]}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                payload = json.load(r)
            if "choices" not in payload:
                raise RuntimeError(f"no choices: {str(payload)[:300]}")
            return payload
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or e.code >= 500:
                time.sleep(min(60, 4 * 2 ** attempt)); continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:500]!r}")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 4 * 2 ** attempt))
    raise RuntimeError(f"8 consecutive failures; last={last!r}")


# ---------------------------------------------------------------- one run
def run_task(task):
    model, recipe_root, domain, category, level, seed, out_path, work_root, hint, key_env, name_botorch = task
    out_path = Path(out_path)
    if out_path.exists():
        return str(out_path), "skip"
    try:
        from run_noise_sweep import Logged, _setup
        fb, df, M, target, mask, objective, norm, x0, selected, recipe0 = _setup(domain, category, recipe_root)
        sigma = level * norm
        noise_seed = int(3_000_017 + 10_007 * seed + int(level * 1000))
        names = [str(n) for n in df["ingredient"]]
        obj = Logged(objective, sigma, noise_seed, names)
        workdir = Path(tempfile.mkdtemp(prefix=f"{category.replace(' ', '_')}_", dir=work_root)).resolve()

        history = []

        def record(recipe):
            history.append({"iteration": len(obj.obs), "recipe": recipe, "loss": obj.obs[-1]})
            (workdir / "evaluations.json").write_text(json.dumps(
                [{"iteration": h["iteration"], "recipe": h["recipe"], "target_dissimilarity": h["loss"]}
                 for h in history], separators=(",", ":")))

        obj(x0)
        record(recipe0)
        messages = [{"role": "system", "content": fb._LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": task_prompt(fb, df, category, history, hint, name_botorch)}]
        calls, cost, nudges, n_py, events = [], 0.0, 0, 0, []
        recommended_iteration = None
        for turn in range(MAX_TURNS):
            done = len(obj.obs) >= BUDGET
            payload = chat(fb, model, messages, TOOLS, tool_choice="none" if done else "auto", key_env=key_env)
            u = payload.get("usage") or {}
            cost += float(u.get("cost") or 0.0)
            calls.append({k: u.get(k) for k in ("prompt_tokens", "completion_tokens", "cost")})
            msg = payload["choices"][0]["message"]
            keep = {k: msg[k] for k in ("role", "content", "tool_calls", "reasoning_details") if msg.get(k) is not None}
            keep.setdefault("role", "assistant"); keep.setdefault("content", "")
            messages.append(keep)
            if done:
                m = re.search(r"\d+", msg.get("content") or "")
                recommended_iteration = int(m.group(0)) if m else None
                break
            tcs = msg.get("tool_calls") or []
            for tc in tcs:
                fn = tc.get("function", {})
                name = fn.get("name")
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError as e:
                    args, res = None, f"Invalid JSON arguments: {e}"
                if args is not None and name == "evaluate":
                    if len(obj.obs) >= BUDGET:
                        res = "Evaluation budget exhausted; no evaluation was made."
                    else:
                        try:
                            x, recipe = fb._parse_recipe(json.dumps({"recipe": args.get("recipe")}), df,
                                                         max_ingredients=MAX_ING)
                            obj(x)
                            record(recipe)
                            res = json.dumps({"iteration": len(obj.obs), "target_dissimilarity": obj.obs[-1],
                                              "evaluations_remaining": BUDGET - len(obj.obs)})
                        except Exception as e:  # noqa: BLE001
                            res = f"Invalid formulation (no evaluation used): {e}"
                    events.append({"turn": turn, "tool": "evaluate", "n_evals": len(obj.obs)})
                elif args is not None and name == "python":
                    n_py += 1
                    res = run_python(str(args.get("code", "")), workdir, n_py)
                    events.append({"turn": turn, "tool": "python", "n_evals": len(obj.obs)})
                elif args is not None:
                    res = f"Unknown tool {name!r}."
                messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": res})
            if len(obj.obs) >= BUDGET:
                messages.append({"role": "user", "content":
                                 f"All {BUDGET} evaluations have been used. Which evaluated formulation do you "
                                 "recommend as the final formulation? Reply with only its iteration number."})
            elif not tcs:
                if nudges >= MAX_NUDGES:
                    break
                nudges += 1
                messages.append({"role": "user", "content":
                                 f"You have {BUDGET - len(obj.obs)} evaluations remaining. Continue."})
            if cost > 8.0:
                events.append({"turn": turn, "abort": "cost cap"})
                break
        n = len(obj.obs)
        rec_true, best_i = [], 0
        for t in range(n):
            if obj.obs[t] < obj.obs[best_i]:
                best_i = t
            rec_true.append(obj.true[best_i])
        own = (obj.true[recommended_iteration - 1]
               if recommended_iteration and 1 <= recommended_iteration <= n else None)
        out = dict(model=model, domain=domain, category=category, level=level, sigma=sigma, norm=norm,
                   arm=("agent_botorch" if name_botorch else "agent_hint" if hint else "agent"), hint=bool(hint), name_botorch=bool(name_botorch), seed=seed, noise_seed=noise_seed, n_evals=n, complete=n >= BUDGET,
                   true_loss=obj.true, observed_loss=obj.obs, recommended_true_loss=rec_true,
                   best_true_so_far=list(np.minimum.accumulate(obj.true)), recipes=obj.recipes,
                   recommended_iteration=recommended_iteration, own_pick_true_loss=own,
                   n_api_calls=len(calls), n_python_calls=n_py, nudges=nudges, cost_usd=cost, events=events,
                   python_code=[(workdir / f"_cell{i:03d}.py").read_text() for i in range(1, n_py + 1)])
        from _run_utils import write_once
        tpath = Path(str(out_path).replace(".json", ".transcript.json"))
        if not write_once(out_path, json.dumps(out),
                          {tpath: json.dumps({"messages": messages, "calls": calls}, indent=0)}):
            return str(out_path), "skip (result already written by another run)"
        return str(out_path), f"ok evals={n} py={n_py} calls={len(calls)} ${cost:.2f}"
    except Exception:
        return str(out_path), "FAIL\n" + traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--levels", type=float, nargs="+", default=[0.15])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--domains", nargs="+", default=["meat", "dairy"])
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=R / "agent_pilot_v1")
    ap.add_argument("--hint", action="store_true", help="append one neutral sentence about noisy, expensive evaluations")
    ap.add_argument("--name-botorch", action="store_true", help="name BoTorch among the installed libraries")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY", help="environment variable holding the API key")
    ap.add_argument("--work-root", type=Path, default=None, help="sandbox working directories (default: a temp dir)")
    a = ap.parse_args()
    if sys.platform != "darwin":
        raise SystemExit("The python-tool sandbox uses macOS sandbox-exec.")
    if not os.environ.get(a.key_env):
        raise SystemExit(f"Set {a.key_env}")
    work_root = str((a.work_root or Path(tempfile.mkdtemp(prefix="agent_pilot_"))).resolve())
    Path(work_root).mkdir(parents=True, exist_ok=True)
    tasks = []
    for m in a.models:
        for dom in a.domains:
            for c in sorted(json.loads((R / "final_protocol_core_v1" / dom / "summary.json").read_text())):
                if a.categories and c not in a.categories:
                    continue
                for lvl in a.levels:
                    for s in range(a.seeds):
                        p = a.out / m.replace("/", "__") / dom / c.replace(" ", "_") / f"{'agent_botorch' if a.name_botorch else 'agent_hint' if a.hint else 'agent'}_noise{lvl:g}_seed{s}.json"
                        tasks.append((m, MODELS.get(m), dom, c, lvl, s, str(p), work_root, a.hint, a.key_env, a.name_botorch))
    print(f"{len(tasks)} tasks; sandbox work root {work_root}", flush=True)
    done = fails = 0
    from _run_utils import watch_parent
    with ProcessPoolExecutor(max_workers=a.workers, initializer=watch_parent, initargs=(os.getpid(),)) as ex:
        for f in as_completed([ex.submit(run_task, t) for t in tasks]):
            path, status = f.result()
            done += 1; fails += status.startswith("FAIL")
            print(f"[{done}/{len(tasks)}] {status.splitlines()[0]} {path.split('agent_pilot_v1/')[-1]}"
                  + ("\n" + status if status.startswith("FAIL") else ""), flush=True)
    print(f"done: {done} tasks, {fails} failures", flush=True)


if __name__ == "__main__":
    main()
