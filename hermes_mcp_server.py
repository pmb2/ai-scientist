#!/usr/bin/env python3
"""
Hermes MCP Server for SakanaAI AI-Scientist.
Exposes AI Scientist capabilities as MCP tools for Hermes agents.

Default model: qwen/qwen3-coder:free (FREE on OpenRouter, 1M context, code-specialized)

Usage:
  python hermes_mcp_server.py           # starts SSE server on localhost:8030
  python hermes_mcp_server.py --stdio   # stdio mode for Hermes subprocess

Environment:
  OPENROUTER_API_KEY  — for LLM calls via OpenRouter
  AI_SCIENTIST_DIR    — path to AI-Scientist repo (default: this script's dir)
"""

import json
import os
import sys
import subprocess
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Optional
from mcp.server.fastmcp import FastMCP

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.absolute()
AI_SCIENTIST_DIR = Path(os.environ.get("AI_SCIENTIST_DIR", SCRIPT_DIR))
VENV_PYTHON = AI_SCIENTIST_DIR / "venv" / "Scripts" / "python.exe"
TEMPLATES_DIR = AI_SCIENTIST_DIR / "templates"
RESULTS_DIR = AI_SCIENTIST_DIR / "results"

mcp = FastMCP("ai-scientist", log_level="INFO")


def _run_ai_scientist(args: list, timeout: int = 3600) -> dict:
    """Run a command in the AI Scientist venv."""
    cmd = [str(VENV_PYTHON), *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": f"Command timed out after {timeout}s",
            "exit_code": -1,
            "success": False,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": str(e),
            "exit_code": -1,
            "success": False,
        }


def _read_json(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _list_templates() -> list:
    """List available experiment templates."""
    templates = []
    if TEMPLATES_DIR.exists():
        for d in sorted(TEMPLATES_DIR.iterdir()):
            if d.is_dir() and (d / "experiment.py").exists():
                info = {"name": d.name, "path": str(d)}
                prompt_file = d / "prompt.json"
                if prompt_file.exists():
                    info["description"] = _read_json(prompt_file).get(
                        "system", ""
                    )[:200]
                templates.append(info)
    return templates


def _list_results(experiment: str = None) -> list:
    """List completed results."""
    results = []
    base = RESULTS_DIR
    if experiment:
        base = base / experiment
    if base.exists():
        for d in sorted(base.iterdir()):
            if d.is_dir():
                info = {"name": d.name, "path": str(d), "experiment": experiment or "unknown"}
                review_file = d / "review.txt"
                if review_file.exists():
                    info["reviewed"] = True
                pdf_files = list(d.glob("*.pdf"))
                if pdf_files:
                    info["papers"] = [p.name for p in pdf_files]
                results.append(info)
    return results


# ============================================================
# MCP Tools
# ============================================================


@mcp.tool()
def list_templates() -> str:
    """List available experiment templates (nanoGPT, grokking, 2d_diffusion, etc.)."""
    templates = _list_templates()
    if not templates:
        return "No templates found. Check AI_SCIENTIST_DIR or re-clone the repo."
    lines = ["**Available Templates:**\n"]
    for t in templates:
        desc = t.get("description", "")
        lines.append(f"- **{t['name']}**: {desc[:150]}")
    return "\n".join(lines)


@mcp.tool()
def list_results(experiment: str = "") -> str:
    """List completed experiment results. Optionally filter by experiment name."""
    results = _list_results(experiment or None)
    if not results:
        return "No results found yet." + (f" for experiment '{experiment}'" if experiment else "")
    lines = ["**Completed Results:**\n"]
    for r in results:
        status = "✅ reviewed" if r.get("reviewed") else "⏳ unreviewed"
        papers = f", papers: {', '.join(r.get('papers', []))}" if r.get("papers") else ""
        lines.append(f"- **{r['name']}** ({r['experiment']}) — {status}{papers}")
    return "\n".join(lines)


@mcp.tool()
def generate_ideas(
    experiment: str,
    model: str = "openrouter/qwen/qwen3-coder:free",
    num_ideas: int = 5,
    num_reflections: int = 3,
    skip_novelty_check: bool = False,
) -> str:
    """
    Generate novel research ideas for an experiment template.
    
    Args:
        experiment: Template name (e.g., 'nanoGPT_lite', 'grokking', '2d_diffusion')
        model: Model identifier — prefix with 'openrouter/' for OpenRouter models
               (e.g., 'openrouter/google/gemma-4-31b-it:free')
        num_ideas: Number of ideas to generate (default: 5)
        num_reflections: Number of reflection rounds per idea (default: 3)
        skip_novelty_check: Skip Semantic Scholar novelty check (default: False)
    """
    # Validate experiment exists
    templates = _list_templates()
    if not any(t["name"] == experiment for t in templates):
        available = ", ".join(t["name"] for t in templates)
        return f"❌ Template '{experiment}' not found. Available: {available}"

    args = [
        "-m", "launch_scientist",
        "--experiment", experiment,
        "--model", model,
        "--num-ideas", str(num_ideas),
        "--skip-experiments",
    ]
    
    if skip_novelty_check:
        args.append("--skip-novelty-check")

    # We modify launch_scientist to accept --skip-experiments, but for now
    # we just do idea generation directly via the generate_ideas module
    cmd = [
        str(VENV_PYTHON), "-c",
        f"""
import sys
sys.path.insert(0, r'{AI_SCIENTIST_DIR}')
import json
from ai_scientist.llm import create_client
from ai_scientist.generate_ideas import generate_ideas as _gen, check_idea_novelty

client, client_model = create_client("{model}")
base_dir = r'{AI_SCIENTIST_DIR}/templates/{experiment}'

ideas = _gen(
    base_dir,
    client=client,
    model=client_model,
    skip_generation=False,
    max_num_generations={num_ideas},
    num_reflections={num_reflections},
)
print(json.dumps(ideas, indent=2))
"""
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ},
        )
        output = result.stdout
        error = result.stderr
        
        if result.returncode != 0:
            return f"❌ Idea generation failed:\n{error[:1000]}"
        
        # Try to parse JSON from output
        try:
            ideas = json.loads(output)
        except json.JSONDecodeError:
            return f"Raw output:\n{output[:3000]}"
        
        lines = [f"**Generated {len(ideas)} Ideas:**\n"]
        for i, idea in enumerate(ideas):
            name = idea.get("Name", f"Idea {i+1}")
            title = idea.get("Title", "No title")
            novel = "✅" if idea.get("novel", True) else "❌"
            lines.append(f"### {novel} {name}")
            lines.append(f"**Title:** {title}")
            exp = idea.get("Experiment", "")
            if exp:
                lines.append(f"**Experiment:** {exp[:200]}")
            lines.append("")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


@mcp.tool()
def run_experiment(
    experiment: str,
    model: str = "openrouter/qwen/qwen3-coder:free",
    num_ideas: int = 1,
    improvement: bool = False,
    gpu_id: str = "0",
) -> str:
    """
    Run a full AI Scientist experiment pipeline (ideas → experiments → paper → review).
    
    This runs launch_scientist.py with the specified template and model.
    Expects baseline runs to be prepared first (see prepare_baseline).
    
    Args:
        experiment: Template name (e.g., 'nanoGPT_lite', 'grokking')
        model: Model to use — prefix with 'openrouter/' for OpenRouter
        num_ideas: Number of ideas to process (default: 1)
        improvement: Run improvement pass after review (default: False)
        gpu_id: GPU device ID to use (default: '0')
    """
    templates = _list_templates()
    if not any(t["name"] == experiment for t in templates):
        available = ", ".join(t["name"] for t in templates)
        return f"❌ Template '{experiment}' not found. Available: {available}"

    import threading
    
    results_log = []
    
    def run():
        args = [
            "-m", "launch_scientist",
            "--experiment", experiment,
            "--model", model,
            "--num-ideas", str(num_ideas),
        ]
        if improvement:
            args.append("--improvement")
        if gpu_id:
            args.extend(["--gpus", gpu_id])
        
        cmd = [str(VENV_PYTHON)] + args
        proc = subprocess.Popen(
            cmd,
            cwd=str(AI_SCIENTIST_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id},
        )
        for line in iter(proc.stdout.readline, ""):
            if line:
                results_log.append(line.rstrip())
        proc.wait()
        results_log.append(f"EXIT_CODE: {proc.returncode}")
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=5)
    
    return (
        f"🚀 **AI Scientist started** for template '{experiment}' with model '{model}'.\n"
        f"Running in background — check results in `{RESULTS_DIR / experiment}`.\n"
        f"Use `list_results(experiment='{experiment}')` to check progress.\n\n"
        f"Initial log:\n" + "\n".join(results_log[-20:])
    )


@mcp.tool()
def prepare_baseline(experiment: str, gpu_id: str = "0") -> str:
    """
    Prepare baseline experiment runs needed before running AI Scientist.
    Required first step for any template.
    
    Args:
        experiment: Template name (e.g., 'nanoGPT_lite', 'grokking', '2d_diffusion')
        gpu_id: GPU device ID (default: '0')
    """
    template_dir = TEMPLATES_DIR / experiment
    if not (template_dir / "experiment.py").exists():
        available = ", ".join(t["name"] for t in _list_templates())
        return f"❌ Template '{experiment}' not found. Available: {available}"
    
    # Handle data preparation for specific templates
    data_steps = []
    if experiment == "nanoGPT_lite":
        # Data already prepared in repo
        pass
    elif experiment == "nanoGPT":
        data_steps = [
            f"{str(VENV_PYTHON)} data/enwik8/prepare.py",
            f"{str(VENV_PYTHON)} data/shakespeare_char/prepare.py",
            f"{str(VENV_PYTHON)} data/text8/prepare.py",
        ]
    
    cmd = [str(VENV_PYTHON), "experiment.py", "--out_dir", "run_0"]
    
    output_lines = []
    
    # Run data preparation
    for ds in data_steps:
        output_lines.append(f"📊 Preparing data: {ds}")
        r = subprocess.run(
            ds,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True, text=True,
            shell=True,
            timeout=600,
        )
        output_lines.append(r.stdout[-500:] if r.stdout else "done")
        if r.returncode != 0:
            output_lines.append(f"⚠️ Data prep warning: {r.stderr[:200]}")
    
    # Run baseline experiment
    output_lines.append(f"\n🔬 Running baseline experiment for '{experiment}'...")
    output_lines.append(f"   {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(template_dir),
            capture_output=True,
            text=True,
            timeout=3600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id},
        )
        output_lines.append(result.stdout[-2000:] if result.stdout else "")
        if result.returncode != 0:
            output_lines.append(f"⚠️ Stderr: {result.stderr[-500:]}")
        else:
            output_lines.append("✅ Baseline run completed!")
            # Run plot script
            plot_cmd = [str(VENV_PYTHON), "plot.py"]
            plot_result = subprocess.run(
                plot_cmd,
                cwd=str(template_dir),
                capture_output=True, text=True,
                timeout=300,
                env={**os.environ},
            )
            if plot_result.returncode == 0:
                output_lines.append("✅ Plot generation completed!")
            else:
                output_lines.append(f"⚠️ Plot warning: {plot_result.stderr[:200]}")
    except subprocess.TimeoutExpired:
        output_lines.append("❌ Baseline experiment timed out (1h limit)")
    except Exception as e:
        output_lines.append(f"❌ Error: {e}")
    
    return "\n".join(output_lines)


@mcp.tool()
def review_paper(pdf_path: str, model: str = "openrouter/nousresearch/hermes-3-llama-3.1-405b:free") -> str:
    """
    Review a generated paper PDF using an LLM.
    
    Args:
        pdf_path: Path to the PDF file to review
        model: Model to use for review (default: gpt-4o-mini)
    """
    pdf = Path(pdf_path)
    if not pdf.exists():
        return f"❌ PDF not found: {pdf_path}"
    
    cmd = [
        str(VENV_PYTHON), "-c",
        f"""
import sys, os
sys.path.insert(0, r'{AI_SCIENTIST_DIR}')
import json
from ai_scientist.perform_review import load_paper, perform_review
import openai

paper_txt = load_paper(r'{pdf}')
client = openai.OpenAI(
    api_key=os.environ.get('OPENROUTER_API_KEY', os.environ.get('OPENAI_API_KEY', '')),
    base_url='https://openrouter.ai/api/v1',
)
review = perform_review(
    paper_txt,
    model='{model}',
    client=client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)
print(json.dumps(review, indent=2))
"""
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ},
        )
        if result.returncode != 0:
            return f"❌ Review failed:\n{result.stderr[:1000]}"
        
        try:
            review = json.loads(result.stdout)
        except json.JSONDecodeError:
            return f"Raw output:\n{result.stdout[:3000]}"
        
        overall = review.get("Overall", "N/A")
        decision = review.get("Decision", "N/A")
        weaknesses = review.get("Weaknesses", [])
        
        lines = [
            f"**Paper Review: {pdf.name}**",
            f"**Overall Score:** {overall}/10",
            f"**Decision:** {decision}",
            "",
        ]
        if weaknesses:
            lines.append("**Weaknesses:**")
            for w in weaknesses[:5]:
                lines.append(f"- {w[:200]}")
            lines.append("")
        
        summary = review.get("Summary", "")
        if summary:
            lines.append(f"**Summary:** {summary[:500]}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {e}"


@mcp.tool()
def system_status() -> str:
    """Check AI Scientist system status: GPU, venv, templates."""
    lines = ["**AI Scientist System Status:**\n"]
    
    # GPU
    gpu_check = subprocess.run(
        [str(VENV_PYTHON), "-c", "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"],
        capture_output=True, text=True, timeout=10,
    )
    gpu_out = gpu_check.stdout.strip().split("\n")
    cuda_avail = gpu_out[0] == "True" if gpu_out else False
    gpu_count = gpu_out[1] if len(gpu_out) > 1 else "0"
    
    lines.append(f"**GPU Available:** {'✅' if cuda_avail else '❌'} ({gpu_count})")
    
    # Venv
    venv_ok = VENV_PYTHON.exists()
    lines.append(f"**Venv:** {'✅' if venv_ok else '❌'} at {VENV_PYTHON}")
    
    # Templates
    templates = _list_templates()
    lines.append(f"**Templates:** {len(templates)} found")
    for t in templates[:5]:
        lines.append(f"  - {t['name']}")
    if len(templates) > 5:
        lines.append(f"  ... and {len(templates) - 5} more")
    
    # Results
    results = _list_results()
    lines.append(f"**Completed Results:** {len(results)}")
    
    # OpenRouter key
    or_key = os.environ.get("OPENROUTER_API_KEY", "")
    lines.append(f"**OpenRouter API Key:** {'✅ set' if or_key else '❌ not set'}")
    
    return "\n".join(lines)


@mcp.tool()
def read_template_info(experiment: str) -> str:
    """Read template description, seed ideas, and prompt info for an experiment."""
    template_dir = TEMPLATES_DIR / experiment
    if not template_dir.exists():
        available = ", ".join(t["name"] for t in _list_templates())
        return f"❌ Template '{experiment}' not found. Available: {available}"
    
    parts = [f"**Template: {experiment}**\n"]
    
    # prompt.json
    prompt_file = template_dir / "prompt.json"
    if prompt_file.exists():
        prompt = _read_json(prompt_file)
        parts.append("**Description:**")
        parts.append(prompt.get("system", "N/A")[:500])
        parts.append("")
    
    # seed_ideas.json
    seed_file = template_dir / "seed_ideas.json"
    if seed_file.exists():
        seeds = _read_json(seed_file)
        if isinstance(seeds, list):
            parts.append(f"**Seed Ideas:** {len(seeds)}")
            for s in seeds[:3]:
                parts.append(f"- {s.get('Name', 'Idea')}: {s.get('Title', '')[:100]}")
            if len(seeds) > 3:
                parts.append(f"  ... and {len(seeds)-3} more")
    
    # README if exists
    readme = template_dir / "README.md"
    if readme.exists():
        parts.append("**README:** (check the template folder for full details)")
    
    return "\n".join(parts)


@mcp.tool()
def run_command(command: str, timeout: int = 300) -> str:
    """
    Run an arbitrary command in the AI Scientist venv.
    Useful for data preparation scripts, debugging, etc.
    
    Args:
        command: Shell command to run (e.g., 'python data/enwik8/prepare.py')
        timeout: Timeout in seconds (default: 300)
    """
    try:
        result = subprocess.run(
            command,
            cwd=str(AI_SCIENTIST_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
        )
        output = result.stdout[-3000:] if result.stdout else ""
        error = result.stderr[-1000:] if result.stderr else ""
        exit_code = result.returncode
        
        parts = [f"**Exit Code:** {exit_code}"]
        if output:
            parts.append(f"\n**Output:**\n{output}")
        if error:
            parts.append(f"\n**Stderr:**\n{error}")
        
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {timeout}s"
    except Exception as e:
        return f"❌ Error: {e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdio", action="store_true", help="Use stdio transport")
    args = parser.parse_args()
    
    if args.stdio:
        mcp.run(transport="stdio")
    else:
        print("Starting AI Scientist MCP server on http://localhost:8030/sse")
        mcp.run(transport="sse", host="localhost", port=8030)
