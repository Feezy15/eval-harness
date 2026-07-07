"""Prompt files: the instrument definitions (effort levels, judge rubric).

Prompt text lives in this package — versioned by git like any code — and its
identity is a content fingerprint stamped into every episode record. A version
string alone can drift from the text it claims to describe (edit the prompt,
forget the bump); the hash cannot, so results produced by different prompt
text can never claim the same instrument.
"""

import hashlib
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def load_prompt(name: str) -> str:
    """Return the text of ``<name>.md``, stripped of surrounding whitespace."""
    path = _PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        available = sorted(p.stem for p in _PROMPTS_DIR.glob("*.md"))
        raise ValueError(f"Unknown prompt {name!r}; available: {available}")
    return path.read_text().strip()


def prompts_fingerprint(root: Path | None = None) -> str:
    """Content hash over every prompt file (name + text), first 12 hex chars.

    File names are part of the identity: the same text moved to a different
    prompt slot is a different instrument, so a rename changes the fingerprint.
    """
    digest = hashlib.sha256()
    for path in sorted((root if root is not None else _PROMPTS_DIR).glob("*.md")):
        digest.update(path.name.encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x00")
    return digest.hexdigest()[:12]


# Snapshotted at import — the same moment consumers snapshot the prompt *text*
# (e.g. the effort-prompt table). Recomputing from disk per episode would let a
# file edited mid-run stamp records with a hash describing text no episode
# actually used.
PROMPTS_FINGERPRINT = prompts_fingerprint()
