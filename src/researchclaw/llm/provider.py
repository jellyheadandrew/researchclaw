from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from researchclaw.config import ResearchClawConfig

_ONBOARD_HINT = (
    "Run 'researchclaw' to set up a provider via onboarding, "
    "or install the SDK manually."
)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        """Send messages to the LLM and return the response.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            system: Optional system prompt.

        Returns:
            The LLM's response text.
        """
        ...

    @abstractmethod
    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        """Send messages to the LLM and yield response chunks.

        Args:
            messages: List of dicts with 'role' and 'content' keys.
            system: Optional system prompt.

        Yields:
            Chunks of the LLM's response text.
        """
        ...


class ClaudeCLIProvider(LLMProvider):
    """Tier 0 provider: shells out to 'claude -p' via subprocess.

    Zero extra packages required — uses claude CLI already on user's system.
    """

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self.model = model

    def _build_prompt(self, messages: list[dict[str, str]], system: str = "") -> str:
        """Format messages and system prompt into a single text prompt for claude -p."""
        parts: list[str] = []
        if system:
            parts.append(f"[System]\n{system}\n")
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        return "\n\n".join(parts)

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        """Send messages to claude CLI via subprocess and return response."""
        prompt = self._build_prompt(messages, system)
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Please install it or run 'researchclaw' "
                "to set up an alternative provider via onboarding."
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("Claude CLI timed out after 120 seconds.")

        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {stderr}")

        return result.stdout.strip()

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        """Stream response from claude CLI via subprocess.

        Yields lines of output as they become available.
        """
        prompt = self._build_prompt(messages, system)
        try:
            proc = subprocess.Popen(
                ["claude", "-p", "--model", self.model],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Please install it or run 'researchclaw' "
                "to set up an alternative provider via onboarding."
            )

        assert proc.stdin is not None
        assert proc.stdout is not None

        proc.stdin.write(prompt)
        proc.stdin.close()

        for line in proc.stdout:
            yield line

        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(
                f"Claude CLI failed (exit {proc.returncode}): {stderr.strip()}"
            )


class ClaudeAgentSDKProvider(LLMProvider):
    """Tier 1 provider: uses claude-agent-sdk Python package.

    Provides richer integration than raw CLI, with system prompts,
    allowed tools, and hooks via the SDK's query() interface.
    """

    def __init__(self, model: str = "claude-opus-4-6") -> None:
        self.model = model
        self._sdk: Any = None

    def _get_sdk(self) -> Any:
        """Lazily import claude_agent_sdk. Raises RuntimeError if not installed."""
        if self._sdk is not None:
            return self._sdk
        try:
            import claude_agent_sdk  # type: ignore[import-untyped]

            self._sdk = claude_agent_sdk
            return self._sdk
        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk is not installed. " + _ONBOARD_HINT
            )

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        sdk = self._get_sdk()
        result = sdk.query(
            prompt=messages[-1].get("content", "") if messages else "",
            system=system,
            model=self.model,
        )
        return str(result)

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        # claude-agent-sdk doesn't support streaming natively; fall back to non-streaming
        yield self.chat(messages, system)


class AnthropicProvider(LLMProvider):
    """Tier 2 provider: uses the anthropic Python SDK with an API key.

    Direct API access — requires the user's Anthropic API key.
    """

    def __init__(self, api_key: str, model: str = "claude-opus-4-6") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily import anthropic and create a client."""
        if self._client is not None:
            return self._client
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "anthropic SDK is not installed. " + _ONBOARD_HINT
            )
        self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        # Extract text from content blocks
        parts: list[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        with client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text


class OpenAIProvider(LLMProvider):
    """Tier 2 provider: uses the openai Python SDK with an API key.

    Direct API access — requires the user's OpenAI API key.
    """

    def __init__(self, api_key: str, model: str = "gpt-4") -> None:
        self.api_key = api_key
        self.model = model
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily import openai and create a client."""
        if self._client is not None:
            return self._client
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError:
            raise RuntimeError(
                "openai SDK is not installed. " + _ONBOARD_HINT
            )
        self._client = openai.OpenAI(api_key=self.api_key)
        return self._client

    def chat(self, messages: list[dict[str, str]], system: str = "") -> str:
        client = self._get_client()
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)
        response = client.chat.completions.create(
            model=self.model,
            messages=api_messages,
        )
        choice = response.choices[0]
        return choice.message.content or ""

    def chat_stream(self, messages: list[dict[str, str]], system: str = "") -> Iterator[str]:
        client = self._get_client()
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})
        api_messages.extend(messages)
        response = client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            stream=True,
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


def get_provider(config: ResearchClawConfig, **kwargs: Any) -> LLMProvider:
    """Factory function that returns the appropriate LLM provider based on config.

    Selects provider based on config.provider:
    - 'claude_cli' (default): Tier 0 — shells out to claude CLI
    - 'claude_agent_sdk': Tier 1 — uses claude-agent-sdk Python package
    - 'anthropic': Tier 2 — uses anthropic SDK with API key
    - 'openai': Tier 2 — uses openai SDK with API key
    """
    provider = config.provider

    if provider == "claude_agent_sdk":
        return ClaudeAgentSDKProvider(model=config.model)
    elif provider == "anthropic":
        if not config.api_key:
            raise ValueError(
                "Anthropic provider requires an API key. "
                "Set api_key in config or run onboarding."
            )
        return AnthropicProvider(api_key=config.api_key, model=config.model)
    elif provider == "openai":
        if not config.api_key:
            raise ValueError(
                "OpenAI provider requires an API key. "
                "Set api_key in config or run onboarding."
            )
        return OpenAIProvider(api_key=config.api_key, model=config.model)
    else:
        # Default: Tier 0 claude CLI
        return ClaudeCLIProvider(model=config.model)
