#!/usr/bin/env python3
"""Compile .pi/agents/*.md into .pi/settings.json for @d3ara1n/pi-subagent.

Source of truth: .pi/agents/*.md (frontmatter + system-prompt body).
Generated artifact: .pi/settings.json (modelRoles + subagent.agentOverrides).

Frontmatter schema (all keys except `name` map 1:1 onto pi-subagent's
agentOverrides role fields — see ~/.pi/agent/npm/node_modules/@d3ara1n/pi-subagent/README.md):

    name:            role key used in subagent_delegate(role=...)   [must match filename]
    description:     one-liner fed to the LLM                       [required]
    decisionTrigger: "when should the model pick this role?"         [required]
    examples:        list of example tasks                           [required, >=1]
    modelRole:       key into MODEL_ROLES below                      [required]
    tools:           list of exact tool names; absent = ALL tools    [optional]
    subagentRoles:   roles this role may spawn                       [optional]
    timeout:         seconds; 0 = unlimited                          [optional, default 0]

Body after the closing `---` is the systemPrompt, verbatim.

Usage: python3 scripts/build_pi_agents.py [--check]
  --check: validate only, exit 1 if .pi/settings.json is stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / ".pi" / "agents"
SETTINGS = ROOT / ".pi" / "settings.json"

# modelRole -> pi-model-roles role definition (provider/model id as registered in pi).
MODEL_ROLES: dict[str, dict] = {
    # main driver: strongest GLM, used by the overseer (and the project default model)
    "glm-5.3": {"model": "zai/glm-5.3"},
    # workhorse: every pipeline/builder/reviewer agent
    "glm-5.3-flash": {"model": "zai/glm-5.3-flash"},
    # independent reviewer models (dual-review setup vs GLM)
    "deepseek-v4-flash": {"model": "openrouter/deepseek/deepseek-v4-flash-0731"},
}

# Project-default main model ("main driver") pinned in project settings.
DEFAULT_PROVIDER = "zai"
DEFAULT_MODEL = "glm-5.3"

ALLOWED_KEYS = {
    "name", "description", "decisionTrigger", "examples",
    "modelRole", "tools", "subagentRoles", "timeout",
}
REQUIRED_KEYS = {"name", "description", "decisionTrigger", "examples", "modelRole"}


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def load_agents() -> dict[str, dict]:
    overrides: dict[str, dict] = {}
    for path in sorted(AGENTS_DIR.glob("*.md")):
        raw = path.read_text()
        if not raw.startswith("---\n"):
            fail(f"{path.name}: missing frontmatter")
        try:
            fm_raw, body = raw[4:].split("\n---\n", 1)
        except ValueError:
            fail(f"{path.name}: no closing '---' for frontmatter")
        fm = yaml.safe_load(fm_raw)
        if not isinstance(fm, dict):
            fail(f"{path.name}: frontmatter is not a mapping")

        unknown = set(fm) - ALLOWED_KEYS
        if unknown:
            fail(f"{path.name}: unknown frontmatter keys {sorted(unknown)}")
        missing = REQUIRED_KEYS - set(fm)
        if missing:
            fail(f"{path.name}: missing required keys {sorted(missing)}")
        if fm["name"] != path.stem:
            fail(f"{path.name}: name {fm['name']!r} != filename stem {path.stem!r}")
        if fm["modelRole"] not in MODEL_ROLES:
            fail(f"{path.name}: modelRole {fm['modelRole']!r} not in MODEL_ROLES "
                 f"({sorted(MODEL_ROLES)}) — add it to MODEL_ROLES in this script")
        if not isinstance(fm["examples"], list) or not fm["examples"]:
            fail(f"{path.name}: examples must be a non-empty list")
        body = body.lstrip("\n")
        if not body.strip():
            fail(f"{path.name}: empty system prompt body")

        role: dict = {
            "role": fm["modelRole"],
            "description": fm["description"],
            "examples": fm["examples"],
            "decisionTrigger": fm["decisionTrigger"],
            "systemPrompt": body,
            "timeout": int(fm.get("timeout", 0)),
        }
        tools = fm.get("tools")
        if isinstance(tools, str):
            tools = [t.strip() for t in tools.split(",") if t.strip()]
        if tools is not None:
            role["tools"] = tools
        if fm.get("subagentRoles"):
            role["subagentRoles"] = fm["subagentRoles"]
        overrides[fm["name"]] = role

    if not overrides:
        fail(f"no agent files found in {AGENTS_DIR}")

    # cross-check subagentRoles references
    for name, role in overrides.items():
        for dep in role.get("subagentRoles", []):
            if dep not in overrides:
                fail(f"{name}: subagentRoles references unknown role {dep!r}")
    return overrides


def build_settings(overrides: dict[str, dict]) -> dict:
    return {
        "defaultProvider": DEFAULT_PROVIDER,
        "defaultModel": DEFAULT_MODEL,
        "modelRoles": {"roles": MODEL_ROLES},
        "subagent": {"agentOverrides": overrides},
    }


def main() -> None:
    check_only = "--check" in sys.argv[1:]
    overrides = load_agents()
    settings = build_settings(overrides)
    text = json.dumps(settings, indent=2, ensure_ascii=False) + "\n"

    if check_only:
        current = SETTINGS.read_text() if SETTINGS.exists() else ""
        if current != text:
            fail(".pi/settings.json is stale — run scripts/build_pi_agents.py")
        print(f"[OK] {len(overrides)} roles, .pi/settings.json up to date")
        return

    SETTINGS.write_text(text)
    print(f"[OK] wrote {SETTINGS} ({len(text)} bytes, {len(overrides)} roles):")
    for name, role in overrides.items():
        tool_note = ",".join(role["tools"]) if "tools" in role else "ALL"
        print(f"  {name:22s} model={role['role']:18s} timeout={role['timeout']:>5}s tools={tool_note}")


if __name__ == "__main__":
    main()
