import asyncio
import os
import subprocess
import sys
from importlib.metadata import version
import re
from urllib.parse import urlparse as _urlparse

from setup import AGENT_NAME, logger

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from azure.ai.agentserver.langgraph import from_langgraph
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, MessagesState, START, END


# ── LLM (Chat Completions API via Azure OpenAI endpoint) ────────────────────

PROJECT_ENDPOINT = os.getenv("AZURE_AI_PROJECT_ENDPOINT")
if not PROJECT_ENDPOINT:
    raise ValueError("AZURE_AI_PROJECT_ENDPOINT environment variable must be set")

MODEL_DEPLOYMENT_NAME = os.getenv("MODEL_DEPLOYMENT_NAME", "gpt-4.1")

_parsed = _urlparse(PROJECT_ENDPOINT)
azure_openai_endpoint = os.environ.get(
    "AZURE_OPENAI_ENDPOINT",
    f"{_parsed.scheme}://{_parsed.netloc}",
)

print(f"Using Azure OpenAI endpoint: {azure_openai_endpoint}")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default",
)

class _AIMessageLogger(BaseCallbackHandler):
    """Log every AI message (content + tool calls) to the console."""

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for gen in generations:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                if msg.content:
                    logger.info(f"AI message: {msg.content}")
                if getattr(msg, "tool_calls", None):
                    logger.info(f"AI tool calls: {msg.tool_calls}")


llm = AzureChatOpenAI(
    model=MODEL_DEPLOYMENT_NAME,
    azure_endpoint=azure_openai_endpoint,
    azure_ad_token_provider=token_provider,
    api_version=os.environ.get("OPENAI_API_VERSION", "2025-03-01-preview"),
    # use_responses_api=True, # there's a bug in response api that will throw error when mcp tool name contains "."
    callbacks=[_AIMessageLogger()],
)

# ── Toolset MCP helpers ────────────────────────────────────────────────────

TOOLSET_ENDPOINT = os.getenv("AZURE_AI_TOOLSET_ENDPOINT")

def _get_toolset_token() -> str:
    """Get bearer token for Toolset MCP endpoint."""
    try:
        credential = DefaultAzureCredential()
        token = credential.get_token("https://ai.azure.com/.default")
        return token.token
    except Exception:
        # Fall back to az CLI
        az_cmd = "az.cmd" if sys.platform == "win32" else "az"
        result = subprocess.run(
            [az_cmd, "account", "get-access-token", "--resource", "https://ai.azure.com", "--query", "accessToken", "-o", "tsv"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get token: {result.stderr}")
        return result.stdout.strip()

def _get_toolset_headers(token: str) -> dict:
    """Get required headers for Toolset MCP calls."""
    return {
        "Authorization": f"Bearer {token}",
        "Foundry-Features": "Toolsets=V1Preview",
    }

# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_auth_urls(exc) -> list[str]:
    """Recursively extract URLs from McpError leaves in an ExceptionGroup."""
    urls: list[str] = []
    if isinstance(exc, ExceptionGroup):
        for sub in exc.exceptions:
            urls.extend(_extract_auth_urls(sub))
    else:
        # Check if this is an McpError whose message is a URL
        msg = str(exc)
        if msg.startswith("http://") or msg.startswith("https://"):
            urls.append(msg.strip())
    return urls

def _log_exception_tree(exc, depth=0):
    """Recursively log nested ExceptionGroup sub-exceptions."""
    prefix = "  " * depth
    if isinstance(exc, ExceptionGroup):
        logger.error(f"{prefix}ExceptionGroup ({len(exc.exceptions)} sub-exceptions): {exc}")
        for i, sub in enumerate(exc.exceptions, 1):
            logger.error(f"{prefix}  [{i}/{len(exc.exceptions)}]:")
            _log_exception_tree(sub, depth + 2)
    else:
        logger.error(f"{prefix}{type(exc).__name__}: {exc}")

# ── Agent creation ──────────────────────────────────────────────────────────

def create_agent(model, tools):
    from langchain.agents import create_agent
    return create_agent(model, tools)

# ── Lazy MCP tool loading ───────────────────────────────────────────────────

_react_agent = None  # cached once tools load successfully

async def _try_load_mcp_tools():
    """Attempt to connect to MCP and load tools.

    Returns (tools, None) on success or (None, error_message) on failure.
    """
    if TOOLSET_ENDPOINT:
        logger.info(f"Connecting to toolset: {TOOLSET_ENDPOINT}")
        token = _get_toolset_token()
        headers = _get_toolset_headers(token)
        client = MultiServerMCPClient(
            {
                "toolset": {
                    "url": TOOLSET_ENDPOINT,
                    "transport": "streamable_http",
                    "headers": headers,
                }
            }
        )
    else:
        client = MultiServerMCPClient(
            {
                "mslearn": {
                    "url": "https://learn.microsoft.com/api/mcp",
                    "transport": "streamable_http",
                }
            }
        )

    try:
        logger.info(f"calling get_tools() on MCP client")
        tools = await client.get_tools()
        logger.info(f"Loaded {len(tools)} tools from MCP")
        return tools, None
    except BaseException as eg:
        _log_exception_tree(eg)
        auth_urls = _extract_auth_urls(eg)
        return None, (auth_urls, str(eg))


async def quickstart():
    """Build and return a LangGraph graph that lazily loads MCP tools.

    - First invocation: tries to load MCP tools.
      If auth fails, returns a message asking the user to authenticate.
    - Subsequent invocations: retries tool loading until it succeeds,
      then caches the react agent and uses it for all future requests.
    """

    async def _try_load_and_run(state: MessagesState):
        global _react_agent

        # If tools haven't loaded yet, try again
        if _react_agent is None:
            tools, error_info = await _try_load_mcp_tools()
            if tools is not None:
                _react_agent = create_agent(llm, tools)
            elif error_info is not None:
                auth_urls, error_msg = error_info
                if auth_urls:
                    url_lines = "\n".join(auth_urls)
                    content = (
                        "Authentication is required to access the MCP toolset. "
                        "Please open the following URL to authenticate, "
                        "then send your request again:\n\n"
                        f"{url_lines}"
                    )
                else:
                    content = (
                        "I couldn't connect to the MCP toolset. "
                        "Please check the configuration and try again.\n\n"
                        f"Error details: {error_msg}"
                    )
                return {
                    "messages": [
                        AIMessage(content=content)
                    ]
                }

        # Invoke the react agent and return its messages.
        # The outer MessagesState's add_messages reducer will deduplicate
        # by message ID — existing messages are updated, new ones appended.
        result = await _react_agent.ainvoke({"messages": state["messages"]})  # type: ignore[arg-type]
        return {"messages": result["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", _try_load_and_run)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile()

async def main():  # pragma: no cover - sample entrypoint
    agent = await quickstart()
    await from_langgraph(agent).run_async()


if __name__ == "__main__":
    asyncio.run(main())