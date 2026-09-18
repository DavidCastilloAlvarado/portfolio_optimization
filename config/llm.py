"""LLM endpoint settings, read from environment variables.

Values come from the environment; the project-root `.env` file (see
`.env.example`) is loaded at import time so machine-specific values
(endpoint, model, MCP server, Chrome) live in one editable file instead of
code. Real environment variables always take precedence over `.env`.

Any OpenAI-compatible chat-completions endpoint is supported:

- LLM_BASE_URL:  e.g. http://192.168.18.200:1234/v1 (OpenAI-compatible root)
- LLM_API_KEY:   optional bearer token (local servers usually need none)
- LLM_MODEL:     model name sent in the request body
- LLM_TIMEOUT:   per-request timeout in seconds (default 120)

When LLM_BASE_URL / LLM_MODEL are unset the settings report
`is_configured == False` and the llm_bl strategy degrades to the pure
equilibrium prior.

Reasoning models are handled in models/blm.py and models/llm_agent.py:
direct view calls disable thinking via chat_template_kwargs unless
LLM_ENABLE_THINKING=1, while the research agent runs with
reasoning_effort (Config.llm_reasoning_effort, default "medium").

The research agent (llm_bl with the agent view engine) consumes a
web-search MCP server over stdio, by default the dsh-google-chrome-search
package (the same chinchilla web-search the developer workstation uses):

- WEBSEARCH_MCP_COMMAND:  stdio command launching the MCP server
                          (unset = agent engine unavailable -> direct engine)
- WEBSEARCH_MCP_ARGS:     its arguments (space-separated)
- CHROME_PATH:            Chrome binary the search server drives
- GSEARCH_PROFILE:        Chrome profile directory holding the Google session
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    timeout: float

    @classmethod
    def from_env(cls) -> "LLMSettings":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "").rstrip("/"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", ""),
            timeout=float(os.environ.get("LLM_TIMEOUT", "120")),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"


@dataclass(frozen=True)
class WebSearchMCPSettings:
    command: str
    args: tuple[str, ...]
    chrome_path: str
    profile_dir: str

    @classmethod
    def from_env(cls) -> "WebSearchMCPSettings":
        return cls(
            command=os.environ.get("WEBSEARCH_MCP_COMMAND", ""),
            args=tuple(os.environ.get("WEBSEARCH_MCP_ARGS", "-y dsh-google-chrome-search mcp").split()),
            chrome_path=os.environ.get("CHROME_PATH", "/usr/bin/google-chrome"),
            profile_dir=os.environ.get("GSEARCH_PROFILE", str(Path.home() / ".dsh-chrome-google")),
        )

    @property
    def is_configured(self) -> bool:
        return bool(self.command)

    def mcp_target(self) -> dict:
        return {
            "mcpServers": {
                "chinchilla": {
                    "type": "stdio",
                    "command": self.command,
                    "args": list(self.args),
                    "env": {
                        "CHROME_PATH": self.chrome_path,
                        "GSEARCH_PROFILE": self.profile_dir,
                    },
                }
            }
        }
