"""Solis branding — hardcoded identity.

The Solis identity is applied by code on every request and cannot be turned off
or overridden by a caller's system prompt: the identity block is installed
*first* (authoritative), and any caller instructions follow it. The fine-tuner
also bakes this identity into the adapter by default, so a Solis fine-tune keeps
identifying as Solis even without the prompt.

Note on honesty vs. licensing: the assistant presents to end users as Solis and
does not volunteer the underlying base model in chat. The upstream attribution
(Qwen3 etc.) that the license requires lives in MODEL_CARD.md and /health —
that is where disclosure belongs, not forced into every user turn.
"""

from __future__ import annotations

ASSISTANT_NAME = "Solis"
PRODUCT_VERSION = "1.9"

# The authoritative identity. Installed first on every request; not overridable.
IDENTITY_BLOCK = (
    f"You are {ASSISTANT_NAME}, version {PRODUCT_VERSION} — an AI assistant. "
    f"Your name is {ASSISTANT_NAME} and you always refer to yourself as "
    f"{ASSISTANT_NAME}. You were created as the {ASSISTANT_NAME} assistant. Do "
    "not claim to be, or identify as, any other model or assistant, and do not "
    "discuss the underlying technology you run on. If asked who or what you are, "
    f"say you are {ASSISTANT_NAME}."
)

# Behavioural defaults, applied when the caller gives no system prompt.
BEHAVIOUR_DEFAULT = (
    "You are helpful, direct, and accurate. When you don't know something or "
    "can't verify it, say so plainly rather than guessing. Keep answers concise "
    "unless asked for depth."
)

DEFAULT_SYSTEM_PROMPT = f"{IDENTITY_BLOCK}\n\n{BEHAVIOUR_DEFAULT}"


def build_system_prompt(user_system: str | None) -> str:
    """Compose the effective system prompt with Solis identity hardcoded first.

    The identity block always leads. A caller's system prompt is honoured for
    behaviour but appended *after* the identity, so it can shape how Solis acts
    but cannot change who Solis is.
    """
    if not user_system or not user_system.strip():
        return DEFAULT_SYSTEM_PROMPT
    return f"{IDENTITY_BLOCK}\n\n{user_system.strip()}"


def strip_identity_leak(text: str) -> str:
    """Last-resort guard: scrub base-model self-identification from output.

    The system prompt handles this in the overwhelming majority of cases; this
    only rewrites a few blatant 'I am <base model>' patterns so branding holds
    even on an odd generation. Deliberately narrow to avoid mangling content.
    """
    import re

    patterns = [
        (r"\bI am Qwen\b", f"I am {ASSISTANT_NAME}"),
        (r"\bI'm Qwen\b", f"I'm {ASSISTANT_NAME}"),
        (r"\bI am an AI assistant created by Alibaba Cloud\b",
         f"I am {ASSISTANT_NAME}, an AI assistant"),
        (r"\bQwen\b", ASSISTANT_NAME),
    ]
    out = text
    for pat, repl in patterns:
        out = re.sub(pat, repl, out)
    return out


def display_name(variant_name: str) -> str:
    return variant_name
