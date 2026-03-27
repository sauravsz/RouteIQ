import os
from typing import Dict, Tuple

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


def _resolve_provider_config() -> Tuple[str, str, str]:
    provider = os.getenv("AI_PROVIDER", "google").strip().lower()

    provider_map = {
        "openai": {
            "key": "OPENAI_API_KEY",
            "base": "OPENAI_BASE_URL",
            "model": "OPENAI_MODEL",
            "default_model": "gpt-4o-mini",
            "default_base": "",
        },
        "groq": {
            "key": "GROQ_API_KEY",
            "base": "GROQ_BASE_URL",
            "model": "GROQ_MODEL",
            "default_model": "openai/gpt-oss-120b",
            "default_base": "https://api.groq.com/openai/v1",
        },
        "cerebras": {
            "key": "CEREBRAS_API_KEY",
            "base": "CEREBRAS_BASE_URL",
            "model": "CEREBRAS_MODEL",
            "default_model": "gpt-oss-120b",
            "default_base": "https://api.cerebras.ai/v1",
        },
        "google": {
            "key": "GOOGLE_API_KEY",
            "base": "GOOGLE_BASE_URL",
            "model": "GOOGLE_MODEL",
            "default_model": "gemini-2.5-flash",
            "default_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        },
    }

    if provider not in provider_map:
        supported = ", ".join(sorted(provider_map.keys()))
        raise RuntimeError(f"Unsupported AI_PROVIDER '{provider}'. Supported values: {supported}.")

    cfg = provider_map[provider]
    api_key = os.getenv(cfg["key"])
    if not api_key:
        raise RuntimeError(
            f"{cfg['key']} not set in environment or .env file for provider '{provider}'."
        )

    model = os.getenv(cfg["model"], cfg["default_model"])
    base_url = os.getenv(cfg["base"], cfg["default_base"])
    return api_key, model, base_url


def build_summary_prompt(summary: Dict, scenario_name: str) -> Tuple[str, str]:
    total_cost = summary["total_cost"]
    factory_util = summary["factory_utilization"]
    warehouse_fill = summary["warehouse_fill_ratio"]

    lines = [
        f"Scenario: {scenario_name}",
        f"Total transportation cost: {total_cost:.2f}",
        "Factory utilization (fraction of capacity used):",
    ]
    for factory, utilization in factory_util.items():
        lines.append(f"  - {factory}: {utilization:.2%}")

    lines.append("Warehouse demand fill ratios:")
    for warehouse, fill_ratio in warehouse_fill.items():
        lines.append(f"  - {warehouse}: {fill_ratio:.2%}")

    bullet_summary = "\n".join(lines)

    system_prompt = (
        "You are a senior supply chain consultant. "
        "Write a clear, non-technical 3-paragraph executive briefing for a COO. "
        "Explain what this transportation optimization solution is doing, "
        "highlight which factories are heavily utilized or underused, "
        "which warehouses are tight on supply, and what the total cost implies. "
        "Avoid formulas, keep it business-focused, and reference the scenario name."
    )

    user_prompt = (
        "Here is a structured summary of the optimized transportation plan:\n\n"
        f"{bullet_summary}\n\n"
        "Write the executive briefing now."
    )

    return system_prompt, user_prompt


def generate_executive_briefing(summary: Dict, scenario_name: str) -> str:
    api_key, model, base_url = _resolve_provider_config()
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt, user_prompt = build_summary_prompt(summary, scenario_name)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("AI API returned an empty briefing.")

    return content