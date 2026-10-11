"""Codex CLI features disabled for every query of the multimodal expert (public version).

Only the list of disabled features is released; scripts/paper_p2_codex_g61.py passes each of them as `--disable <feature>`, so
that the model sees nothing but the prompt and the image pixels.
"""

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
