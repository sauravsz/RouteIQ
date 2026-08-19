import base64
import os
import re
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI, APIError

load_dotenv()

def _get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

PROVIDER_CONFIG = {
    "openai": {
        "key": "OPENAI_API_KEY",
        "base": "OPENAI_BASE_URL",
        "model": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "default_base": "",
        "suggested_models": ["gpt-4o-mini", "gpt-4o"],
    },
    "groq": {
        "key": "GROQ_API_KEY",
        "base": "GROQ_BASE_URL",
        "model": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
        "default_base": "https://api.groq.com/openai/v1",
        "suggested_models": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
    },
    "cerebras": {
        "key": "CEREBRAS_API_KEY",
        "base": "CEREBRAS_BASE_URL",
        "model": "CEREBRAS_MODEL",
        "default_model": "gpt-oss-120b",
        "default_base": "https://api.cerebras.ai/v1",
        "suggested_models": ["gpt-oss-120b"],
    },
    "google": {
        "key": "GOOGLE_API_KEY",
        "base": "GOOGLE_BASE_URL",
        "model": "GOOGLE_MODEL",
        "default_model": "gemini-2.5-flash",
        "default_base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "suggested_models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
    },
}

def get_supported_providers() -> List[str]:
    return list(PROVIDER_CONFIG.keys())

def get_provider_model_options(provider: str) -> List[str]:
    config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["google"])
    configured_model = _get_secret(config["model"], config["default_model"])
    options = list(config["suggested_models"])
    if configured_model not in options:
        options.insert(0, configured_model)
    return options

def get_provider_default_model(provider: str) -> str:
    config = PROVIDER_CONFIG.get(provider, PROVIDER_CONFIG["google"])
    return _get_secret(config["model"], config["default_model"])

def _resolve_provider_config(
    provider_override: str = "",
    model_override: str = "",
    api_key_override: str = "",
) -> Tuple[str, str, str]:
    provider = (provider_override or _get_secret("AI_PROVIDER", "google")).strip().lower()

    if provider not in PROVIDER_CONFIG:
        supported = ", ".join(sorted(PROVIDER_CONFIG.keys()))
        raise RuntimeError(f"Unsupported AI_PROVIDER '{provider}'. Supported values: {supported}.")

    cfg = PROVIDER_CONFIG[provider]
    api_key = (api_key_override or _get_secret(cfg["key"]) or "").strip()
    if not api_key:
        raise RuntimeError(
            f"API Key not provided for provider '{provider}'. Enter it in sidebar or set env '{cfg['key']}'."
        )

    model = (model_override or _get_secret(cfg["model"], cfg["default_model"])).strip()
    base_url = _get_secret(cfg["base"], cfg["default_base"])
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

VISION_MODEL = "gemini-3.5-flash"

def extract_matrix_from_image(
    image_bytes: bytes,
    mime_type: str = "image/png",
    api_key: str = "",
) -> str:
    """Extract cost matrix from image, return long-format CSV string.

    ponytail: vision model does OCR+structure in one shot. Works for clean printed
    tables; hand-drawn or skewed photos may need manual correction after import.
    """
    cfg = PROVIDER_CONFIG["google"]
    resolved_api_key = (api_key or _get_secret(cfg["key"]) or "").strip()
    if not resolved_api_key:
        raise RuntimeError("NO_GOOGLE_API_KEY")
    client = OpenAI(api_key=resolved_api_key, base_url=cfg["default_base"])
    b64 = base64.b64encode(image_bytes).decode("ascii")

    system_prompt = (
        "You extract transportation problem tables from images. "
        "Output ONLY CSV — no markdown fences, no commentary."
    )
    user_prompt = (
        "This image shows a transportation cost matrix with factory supplies and warehouse demands.\n\n"
        "Extract supply, demand, and all route costs. Output long-format CSV with this exact header:\n"
        "scenario,factory,warehouse,supply,demand,cost\n\n"
        "Rules:\n"
        "- scenario is always 'custom'\n"
        "- one row per factory-warehouse pair\n"
        "- supply = that factory's total supply (repeated on each of its rows)\n"
        "- demand = that warehouse's total demand (repeated on each of its rows)\n"
        "- cost = unit shipping cost for that pair\n"
        "- keep factory/warehouse names exactly as shown (e.g. F1, W2 or Factory A, Warehouse B)\n"
        "- numbers only in supply/demand/cost columns, no currency symbols or units"
    )

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ]},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
    except APIError as e:
        raise RuntimeError(f"AI Provider API Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Image extraction failed: {str(e)}")

    content = content.strip()
    content = re.sub(r"^```(?:csv)?\s*|\s*```$", "", content, flags=re.MULTILINE).strip()
    if "scenario,factory,warehouse,supply,demand,cost" not in content:
        raise RuntimeError("Could not extract a valid cost matrix from this image. Try a clearer photo.")
    return content

def generate_executive_briefing(
    summary: Dict,
    scenario_name: str,
    provider: str = "",
    model: str = "",
    api_key: str = "",
) -> str:
    try:
        resolved_api_key, resolved_model, base_url = _resolve_provider_config(
            provider_override=provider,
            model_override=model,
            api_key_override=api_key,
        )
        client = OpenAI(api_key=resolved_api_key, base_url=base_url if base_url else None)

        system_prompt, user_prompt = build_summary_prompt(summary, scenario_name)

        response = client.chat.completions.create(
            model=resolved_model,
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
    except APIError as e:
        raise RuntimeError(f"AI Provider API Error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Executive briefing generation failed: {str(e)}")
