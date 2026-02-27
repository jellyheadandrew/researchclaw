"""
llm.py — LLM abstraction layer.

Supported providers (set in config.yaml under llm.provider):
  - claude_cli  : Claude Code CLI subprocess — uses your existing Claude Code OAuth
                  login. No separate API key needed. Recommended if you use Claude Code.
  - anthropic   : Anthropic Python SDK with ANTHROPIC_API_KEY.
  - openai      : OpenAI GPT-4o / GPT-4 via OPENAI_API_KEY (requires openai>=1.0.0)
  - openrouter  : Multi-model routing via OPENROUTER_API_KEY (requires openai>=1.0.0)
  - deepseek    : DeepSeek-Coder / DeepSeek-R1 via DEEPSEEK_API_KEY (requires openai>=1.0.0)
  - ollama      : Local models via Ollama at http://localhost:11434 (requires openai>=1.0.0)

OpenAI-compatible providers (openai, openrouter, deepseek, ollama) all use the same
OpenAI SDK with different base_url values. Set llm.base_url in config.yaml to override
the default endpoint (useful for Azure OpenAI, vLLM, LM Studio, etc.).
"""

from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .config import Config


# ── Tool-use data structures ─────────────────────────────────────────────────

@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM that may include tool calls."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""


# ── Tool definitions ─────────────────────────────────────────────────────────

AGENT_TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the project. Path is relative to base_dir "
            "(e.g. 'github_codes/train.py', 'sandbox/20260227/trial_001/train.py', "
            "'reference/paper.pdf')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path from base_dir"}},
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and subdirectories in a directory. "
            "Path is relative to base_dir."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Relative path from base_dir"}},
            "required": ["path"],
        },
    },
    {
        "name": "search_files",
        "description": (
            "Search for a text pattern (regex) in files under a directory. "
            "Returns matching lines with file paths and line numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search in (relative to base_dir). Defaults to 'github_codes'."},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file in the current trial's sandbox. "
            "Only available in RESEARCH state. Path is relative to the sandbox directory. "
            "The researcher will be shown a diff and asked to confirm."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path within the sandbox"},
                "content": {"type": "string", "description": "Full file content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command in the sandbox directory. "
            "Only available in RESEARCH state. The researcher will be shown "
            "the command and asked to confirm. Long-running commands will "
            "transition the agent to EXECUTE state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "Shell command to run"}},
            "required": ["command"],
        },
    },
    {
        "name": "propose_action",
        "description": (
            "Propose a lifecycle action. The agent will handle execution "
            "and request researcher confirmation where needed.\n"
            "Actions: start_trial, approve, reject, continue, push"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start_trial", "approve", "reject", "continue", "push"],
                    "description": "The lifecycle action to propose",
                },
                "detail": {
                    "type": "string",
                    "description": "Additional detail (e.g. trial goal for start_trial)",
                },
            },
            "required": ["action"],
        },
    },
]


def _tools_for_anthropic(tools: list[dict]) -> list[dict]:
    """Convert AGENT_TOOLS to Anthropic API tool format."""
    return tools  # Already in Anthropic format


def _tools_for_openai(tools: list[dict]) -> list[dict]:
    """Convert AGENT_TOOLS to OpenAI function-calling format."""
    result = []
    for t in tools:
        result.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        })
    return result


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Implement this interface to add a new backend.
    """

    @property
    def supports_tool_use(self) -> bool:
        """Whether this provider supports native tool/function calling."""
        return False

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        """Send a single-turn completion request and return the response text."""
        ...

    @abstractmethod
    def complete_with_context(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> str:
        """Multi-turn completion with a conversation history in OpenAI message format."""
        ...

    def complete_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Multi-turn completion with tool definitions.
        Returns an LLMResponse that may contain tool_calls.
        Providers that don't support tool use raise NotImplementedError.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support tool use")


class ClaudeOAuthProvider(LLMProvider):
    """
    Uses the `claude` CLI subprocess — inherits Claude Code's OAuth login.
    No separate API key needed. Recommended for users who already use Claude Code.

    Requires Claude Code to be installed: https://claude.ai/code
    The `claude` binary must be available on PATH (or set cli_path in config).
    """

    def __init__(self, model: str = "claude-sonnet-4-6", cli_path: str = "claude"):
        result = subprocess.run(
            [cli_path, "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise EnvironmentError(
                f"claude CLI not found at {cli_path!r}. "
                "Install Claude Code (https://claude.ai/code) and log in first."
            )
        self.model = model
        self.cli_path = cli_path

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        result = subprocess.run(
            [
                self.cli_path, "-p",
                "--system-prompt", system_prompt,
                "--model", self.model,
                user_message,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited with code {result.returncode}:\n{result.stderr}"
            )
        return result.stdout.strip()

    def complete_with_context(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> str:
        # claude -p is non-interactive; serialize history into the user message.
        history_lines = [
            f"[{m['role'].upper()}] {m['content']}"
            for m in messages[:-1]
        ]
        last = messages[-1]["content"] if messages else ""
        if history_lines:
            combined = "Conversation history:\n" + "\n".join(history_lines) + "\n\nUser: " + last
        else:
            combined = last
        return self.complete(system_prompt, combined, max_tokens)


class ClaudeAPIProvider(LLMProvider):
    """
    Uses the Anthropic Claude API.
    Requires the ANTHROPIC_API_KEY environment variable (or the env var named
    in config.llm_api_key_env).
    """

    def __init__(self, model: str = "claude-sonnet-4-6", api_key_env: str = "ANTHROPIC_API_KEY"):
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env!r} is not set. "
                "Export your Anthropic API key before starting the agent."
            )
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is not installed. "
                "Run: pip install anthropic"
            )
        self.model = model

    @property
    def supports_tool_use(self) -> bool:
        return True

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def complete_with_context(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    def complete_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        api_tools = _tools_for_anthropic(tools)
        # Filter messages to Anthropic format (strip any internal keys)
        api_messages = _to_anthropic_messages(messages)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=api_messages,
            tools=api_tools,
        )
        return _parse_anthropic_response(response)


_OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "openai":     "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek":   "https://api.deepseek.com/v1",
    "ollama":     "http://localhost:11434/v1",
}

_OPENAI_COMPAT_KEY_ENVS: dict[str, str] = {
    "openai":     "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "deepseek":   "DEEPSEEK_API_KEY",
    "ollama":     "",   # Ollama does not require a key
}


class OpenAICompatibleProvider(LLMProvider):
    """
    Handles OpenAI, OpenRouter, DeepSeek, and Ollama via the OpenAI-compatible
    REST API.  All four services expose the same /chat/completions endpoint.

    Requires the 'openai' Python package (>= 1.0.0):
        pip install openai

    config.yaml options:
        llm:
          provider: openai          # or openrouter | deepseek | ollama
          model: gpt-4o             # model name appropriate for the provider
          api_key_env: OPENAI_API_KEY   # env var holding the key (uses default if empty)
          base_url: ""              # override URL (empty = provider default)
    """

    def __init__(
        self,
        provider_name: str,
        model: str,
        api_key_env: str = "",
        base_url: str = "",
    ):
        try:
            import openai as _openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is not installed. "
                "Run: pip install openai>=1.0.0"
            )

        # Resolve base URL: explicit config override > provider default
        resolved_url = base_url.strip() or _OPENAI_COMPAT_BASE_URLS.get(provider_name, "")
        if not resolved_url:
            raise ValueError(
                f"No base_url configured for provider {provider_name!r} "
                "and no default is registered."
            )

        # Resolve API key env var: explicit config override > provider default
        key_env = api_key_env.strip() or _OPENAI_COMPAT_KEY_ENVS.get(provider_name, "")
        if key_env:
            api_key = os.environ.get(key_env)
            if not api_key:
                if provider_name == "ollama":
                    api_key = "ollama"  # Ollama ignores the key value
                else:
                    raise EnvironmentError(
                        f"Environment variable {key_env!r} is not set. "
                        f"Export your {provider_name} API key before starting the agent."
                    )
        else:
            api_key = "ollama"  # safe fallback for keyless providers (Ollama)

        self.model = model
        self.provider_name = provider_name
        self._client = _openai.OpenAI(api_key=api_key, base_url=resolved_url)

    @property
    def supports_tool_use(self) -> bool:
        # Ollama may not reliably support function calling
        return self.provider_name != "ollama"

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 4096,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    def complete_with_context(
        self,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int = 4096,
    ) -> str:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
        )
        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int = 4096,
    ) -> LLMResponse:
        oai_tools = _tools_for_openai(tools)
        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages += _to_openai_messages(messages)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=full_messages,
            tools=oai_tools,
        )
        return _parse_openai_response(response)


# ── Message conversion helpers ────────────────────────────────────────────────

def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """
    Convert internal conversation history to Anthropic API message format.
    Handles tool_result messages and strips internal-only keys.
    """
    result = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool_result":
            # Anthropic expects tool results as user messages with tool_result content blocks
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_use_id", ""),
                        "content": m.get("content", ""),
                    }
                ],
            })
        elif role == "assistant" and "tool_calls" in m:
            # Assistant message with tool use — build Anthropic content blocks
            content_blocks = []
            if m.get("text"):
                content_blocks.append({"type": "text", "text": m["text"]})
            for tc in m["tool_calls"]:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.id if isinstance(tc, ToolCall) else tc.get("id", ""),
                    "name": tc.name if isinstance(tc, ToolCall) else tc.get("name", ""),
                    "input": tc.arguments if isinstance(tc, ToolCall) else tc.get("arguments", {}),
                })
            result.append({"role": "assistant", "content": content_blocks})
        else:
            result.append({"role": role, "content": m.get("content", "")})
    return result


def _parse_anthropic_response(response) -> LLMResponse:
    """Parse an Anthropic API response into LLMResponse."""
    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(
                id=block.id,
                name=block.name,
                arguments=block.input if isinstance(block.input, dict) else {},
            ))
    return LLMResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        stop_reason=getattr(response, "stop_reason", ""),
    )


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """
    Convert internal conversation history to OpenAI API message format.
    Handles tool call messages and tool results.
    """
    result = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool_result":
            result.append({
                "role": "tool",
                "tool_call_id": m.get("tool_use_id", ""),
                "content": m.get("content", ""),
            })
        elif role == "assistant" and "tool_calls" in m:
            oai_tool_calls = []
            for tc in m["tool_calls"]:
                name = tc.name if isinstance(tc, ToolCall) else tc.get("name", "")
                args = tc.arguments if isinstance(tc, ToolCall) else tc.get("arguments", {})
                tc_id = tc.id if isinstance(tc, ToolCall) else tc.get("id", "")
                oai_tool_calls.append({
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args) if isinstance(args, dict) else args,
                    },
                })
            msg = {"role": "assistant", "content": m.get("text") or m.get("content") or ""}
            if oai_tool_calls:
                msg["tool_calls"] = oai_tool_calls
            result.append(msg)
        else:
            result.append({"role": role, "content": m.get("content", "")})
    return result


def _parse_openai_response(response) -> LLMResponse:
    """Parse an OpenAI API response into LLMResponse."""
    choice = response.choices[0]
    msg = choice.message
    text = msg.content or ""
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=args,
            ))
    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        stop_reason=getattr(choice, "finish_reason", ""),
    )


def get_llm_provider(config: Config) -> LLMProvider:
    """
    Factory: reads config and returns the appropriate LLM provider.

    config.yaml options:
        llm:
          provider: claude_cli        # Option A — Claude Code OAuth (recommended)
          model: claude-sonnet-4-6
          cli_path: claude            # path to claude binary (default: from PATH)

          provider: anthropic         # Option B — Anthropic API key
          model: claude-sonnet-4-6
          api_key_env: ANTHROPIC_API_KEY

          provider: openai            # Option C — not yet implemented
          provider: ollama            # Option D — not yet implemented
    """
    provider_name = config.llm_provider

    if provider_name == "claude_cli":
        return ClaudeOAuthProvider(
            model=config.llm_model,
            cli_path=config.llm_claude_cli_path,
        )

    elif provider_name == "anthropic":
        return ClaudeAPIProvider(
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
        )

    elif provider_name in ("openai", "openrouter", "deepseek", "ollama"):
        return OpenAICompatibleProvider(
            provider_name=provider_name,
            model=config.llm_model,
            api_key_env=config.llm_api_key_env,
            base_url=config.llm_openai_base_url,
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name!r}. "
            "Supported providers: claude_cli, anthropic, openai, openrouter, deepseek, ollama"
        )
