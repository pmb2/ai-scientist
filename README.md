# AI Scientist — Hermes Agent Integration

**Fork of [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)**
Patched for generic OpenRouter model support + Hermes MCP server.

## What's Here

| File | What |
|------|------|
| `llm.py` | Patched — any OpenRouter model via `provider/model:free` |
| `launch_scientist.py` | Patched — generic Aider model mapping, OpenRouter review step |
| `hermes_mcp_server.py` | **New** — MCP server exposing 9 tools for Hermes agents |
| `.env.example` | Free model config per task (Owl Alpha, Qwen3-Coder, Hermes 405B) |

## Model Map (all FREE on OpenRouter)

| Task | Model |
|------|-------|
| Idea Generation | `openrouter/owl-alpha` (1M ctx) |
| Code/Experiments | `qwen/qwen3-coder:free` (1M ctx) |
| Paper Writing | `openrouter/owl-alpha` (1M ctx) |
| Review | `nousresearch/hermes-3-llama-3.1-405b:free` (131K ctx) |

## MCP Tools

- `system_status` — GPU, venv, templates, API keys
- `list_templates` — available experiment templates
- `list_results` — completed runs
- `generate_ideas` — LLM generates novel research directions
- `run_experiment` — full pipeline: ideas → experiments → paper → review
- `prepare_baseline` — baseline experiment runs
- `review_paper` — LLM peer review
- `read_template_info` — template metadata
- `run_command` — arbitrary venv commands

## Full Repo

The full project with all templates, example papers, and datasets is at:
https://github.com/SakanaAI/AI-Scientist

Clone it, then overlay these patched files.
