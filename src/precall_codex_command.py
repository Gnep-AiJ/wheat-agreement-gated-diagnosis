"""Construct, but never execute, the frozen one-image Codex CLI command."""

from __future__ import annotations

import json
from pathlib import Path


MODEL = "gpt-6-sol"
DISABLED_FEATURES = (
    "apps",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "computer_use",
    "fast_mode",
    "hooks",
    "image_generation",
    "in_app_browser",
    "multi_agent",
    "plugins",
    "remote_plugin",
    "shell_tool",
    "skill_search",
)
CODEX_WSL = "<WSL_HOME>/.local/bin/codex"


def effective_user_prompt(prompt_schema: Path) -> str:
    """Flatten v2 text into the actual single CLI user message.

    `codex exec` does not expose a developer-message flag. The historical
    `developer_prompt` key is content provenance, not its effective role.
    """
    obj = json.loads(prompt_schema.read_text(encoding="utf-8"))
    return f"{obj['developer_prompt']}\n\n{obj['user_prompt']}"


def build_command(*, image_path: Path, schema_path: Path, item_dir: Path,
                  output_path: Path, prompt: str) -> list[str]:
    """Return a fresh, ephemeral, one-image command; no subprocess is run."""
    image_path = image_path.resolve(strict=True)
    schema_path = schema_path.resolve(strict=True)
    item_dir = item_dir.resolve(strict=True)
    output_path = output_path.resolve(strict=False)
    if image_path.parent != item_dir or schema_path.parent != item_dir:
        raise ValueError("image and schema must be inside the current item directory")
    if image_path == schema_path or not prompt.strip():
        raise ValueError("distinct image/schema and a nonempty prompt are required")
    if item_dir in output_path.parents or output_path.exists():
        raise ValueError("new response output must stay outside the item workspace")
    if {path.name for path in item_dir.iterdir()} != {image_path.name, schema_path.name}:
        raise ValueError("item directory must contain exactly image and schema")
    command = [CODEX_WSL, "exec", "--strict-config", "--ephemeral",
               "--ignore-rules",
               "--skip-git-repo-check", "--json", "-m", MODEL,
               "-c", 'model_reasoning_effort="low"',
               "-C", str(item_dir), "--image", str(image_path),
               "--output-schema", str(schema_path),
               "--output-last-message", str(output_path)]
    for feature in DISABLED_FEATURES:
        command.extend(("--disable", feature))
    command.append(prompt)
    return command
