"""
llm.py
------
LLM interpretation helpers for the Smart Green Nudging replication.

All calls go through a single `interpret()` function which falls back
gracefully when Ollama is unavailable.  A `conclude()` shortcut generates
the final synthesis paragraph.

Usage
-----
    from llm import interpret, conclude, print_interpretation
"""

import logging

logger = logging.getLogger(__name__)

_PAPER = "Smart Green Nudging (von Zahn et al., 2024, Marketing Science)"
_RESEARCH_Q = (
    "Do personalised green nudges — shown selectively to customers most "
    "likely to respond — reduce e-commerce product returns while remaining "
    "more profitable than universal nudging?"
)


# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------

def interpret(section_name: str, context: str, sentences: int = 4) -> str:
    """
    Call local Llama3.2 via litellm to interpret one section's results.

    Falls back gracefully if Ollama is not running.

    Parameters
    ----------
    section_name : short label for the notebook section
    context      : pre-formatted string of key results
    sentences    : requested response length (3-5 recommended)

    Returns
    -------
    Plain-English interpretation paragraph (str).
    """
    prompt = (
        f"You are a research assistant helping interpret results from a "
        f"causal inference replication study of '{_PAPER}'.\n\n"
        f"Research question: {_RESEARCH_Q}\n\n"
        f"Section: {section_name}\n\n"
        f"Results summary:\n{context}\n\n"
        f"In {sentences} sentences, interpret these results in the context of "
        f"the paper's research question. Be concise, precise, and highlight "
        f"any notable patterns, significance levels, or concerns."
    )
    return _call_llm(prompt)


def conclude(summary_df) -> str:
    """
    Generate a formal conclusion paragraph from the replication summary table.

    Parameters
    ----------
    summary_df : pd.DataFrame with columns ['Metric', 'Value']

    Returns
    -------
    5-7 sentence academic conclusion (str).
    """
    prompt = (
        f"You are a research assistant writing the conclusion for a "
        f"replication study of '{_PAPER}'.\n\n"
        f"The paper shows that personalised green nudges — shown to customers "
        f"most likely to respond — can reduce e-commerce product returns while "
        f"being more profitable than universal nudging.\n\n"
        f"Replication results:\n{summary_df.to_string(index=False)}\n\n"
        f"Write a 5-7 sentence conclusion paragraph that:\n"
        f"1. States whether the replication supports the paper's main findings.\n"
        f"2. Highlights the most important numerical results.\n"
        f"3. Notes the value of causal-ML personalisation over naive ATE-based policies.\n"
        f"4. Comments on statistical significance and confidence scores.\n"
        f"5. Mentions one limitation of this replication (use of simulated data).\n"
        f"Write in formal academic prose."
    )
    return _call_llm(prompt)


def print_interpretation(section: str, context: str, sentences: int = 4) -> None:
    """Pretty-print an LLM interpretation block to stdout."""
    text = interpret(section, context, sentences=sentences)
    sep  = "═" * 70
    print(f"\n{sep}")
    print(f"  🤖  LLM Interpretation — {section}")
    print(sep)
    print(text)
    print(sep + "\n")


# ---------------------------------------------------------------------------
# Internal: litellm call with graceful fallback
# ---------------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    try:
        import litellm  # optional dependency
        response = litellm.completion(
            model="ollama_chat/llama3.2",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.choices[0].message.content
    except ImportError:
        return "[litellm not installed — run: pip install litellm]"
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return (
            f"[LLM unavailable — start Ollama and run `ollama pull llama3.2`]\n"
            f"Error: {exc}"
        )
