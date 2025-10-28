from __future__ import annotations

import os
from typing import Annotated, Literal, Any

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field
from fastmcp.server.context import Context
from fastmcp.client.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from urllib.parse import urlparse, urlunparse
import json
import time



mcp = FastMCP(
    name="Chainstack MCP PoC",
)

# -----------------------------------------------------------------------------
# Fallback wrapper: expose docs_searchchainstack locally, calling remote MCP
# -----------------------------------------------------------------------------

@mcp.tool(name="docs_searchchainstack")
async def docs_searchchainstack(query: str) -> dict:
    """Search the Chainstack Developer Portal (primary source for Platform docs).

    Use for: features and how‑tos, infrastructure tips, RPS limits, and RPC/REST
    method references. Returns structured results from the Dev Portal MCP.
    """
    # Use explicit Streamable HTTP transport; Mintlify Docs MCP is not SSE
    transport = StreamableHttpTransport("https://docs.chainstack.com/mcp")
    async with Client(
        transport,
        name="DocsProxyClient-call",
    ) as client:
        result = await client.call_tool("SearchChainstack", {"query": query})
        payload: dict[str, Any] = {
            "tool": "SearchChainstack",
            "query": query,
            "is_error": result.is_error,
        }
        if result.structured_content is not None:
            payload["results"] = result.structured_content
        elif result.data is not None:
            payload["results"] = result.data
        else:
            # Collect plain text blocks into an array of entries
            entries: list[dict[str, str]] = []
            current: dict[str, str] = {}
            def flush() -> None:
                nonlocal current
                if current:
                    entries.append(current)
                    current = {}
            for block in result.content:
                text = getattr(block, "text", "") or ""
                text = text.strip()
                if not text:
                    flush();
                    continue
                lowered = text.lower()
                if lowered.startswith("title:"):
                    flush(); current["title"] = text[len("Title:"):].strip(); continue
                if lowered.startswith("link:"):
                    current["link"] = text[len("Link:"):].strip(); continue
                if lowered.startswith("content:"):
                    current.setdefault("content", text[len("Content:"):].strip()); continue
                current.setdefault("content", "")
                if current["content"]:
                    current["content"] += "\n" + text
                else:
                    current["content"] = text
            flush()
            payload["results"] = entries
        return payload

# -----------------------------------------------------------------------------
# Resources
# -----------------------------------------------------------------------------

@mcp.resource("status://components")
async def chainstack_status_components() -> dict:
    """Chainstack service status (Console, API, Docs, networks). Returns JSON {components: [{name, status, group?}]}. Use to answer 'is X up or live or operational etc?'"""
    url = "https://status.chainstack.com/components.json"
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()
    return {
        "resource": "status://components",
        "url": url,
        "mime_type": "application/json",
        "content": data,
    }

# Chainstack Platform API configuration
CHAINSTACK_API_BASE = os.environ.get("CHAINSTACK_API_BASE", "https://api.chainstack.com")
CHAINSTACK_API_KEY_ENV = "CHAINSTACK_PLATFORM_API_KEY"

# Canonicalize common network aliases that users/agents provide
NETWORK_ALIAS_TO_PUBLIC: dict[str, str] = {
    # Ethereum
    "mainnet": "ethereum-mainnet",
    "ethereum": "ethereum-mainnet",
    "eth": "ethereum-mainnet",
    "ethereum-mainnet": "ethereum-mainnet",
    "sepolia": "ethereum-sepolia",
    "ethereum-sepolia": "ethereum-sepolia",
    "holesky": "ethereum-holesky",
    "ethereum-holesky": "ethereum-holesky",
}


# ----------------------------------------------------------------------------
# Endpoint cache (per-session)
# ----------------------------------------------------------------------------

CACHE_TTL_SECONDS = int(os.environ.get("CHAINSTACK_ENDPOINT_CACHE_TTL", "1200"))

# session_id -> {(public_network, endpoint_kind): (created_at, url)}
_SESSION_ENDPOINT_CACHE: dict[str, dict[tuple[str, str], tuple[float, str]]] = {}


def _cache_get(session_id: str, public_network: str, endpoint_kind: str) -> str | None:
    entry = _SESSION_ENDPOINT_CACHE.get(session_id, {}).get((public_network, endpoint_kind))
    if not entry:
        return None
    created_at, url = entry
    if (time.time() - created_at) > CACHE_TTL_SECONDS:
        # Expired
        _SESSION_ENDPOINT_CACHE.get(session_id, {}).pop((public_network, endpoint_kind), None)
        return None
    return url


def _cache_set(session_id: str, public_network: str, endpoint_kind: str, url: str) -> None:
    bucket = _SESSION_ENDPOINT_CACHE.setdefault(session_id, {})
    bucket[(public_network, endpoint_kind)] = (time.time(), url)


class NodeConfiguration(BaseModel):
    """Subset of Ethereum configuration supported in this PoC."""

    archive: bool = Field(default=False, description="Archive flag (must be false for global shared).")
    client: str | None = Field(
        default=None, description="Optional client identifier (reth, geth, etc.)."
    )


class CreateNodeRequest(BaseModel):
    """Payload for provisioning a Chainstack node."""

    name: Annotated[str, Field(min_length=3)]
    network: str
    region: str = "global1"
    provider: str = "chainstack_global"
    role: Literal["peer", "validator"] = "peer"
    type: Literal["shared", "dedicated"] = "shared"
    configuration: NodeConfiguration = Field(default_factory=NodeConfiguration)


class NodeEndpoints(BaseModel):
    """Connection endpoints; credentials are not surfaced.

    When an `auth_key` is provided by the API, the `*_with_key` fields will
    include the ready-to-use URLs with the key appended to the path.
    """

    https_endpoint: str | None = None
    wss_endpoint: str | None = None
    beacon_endpoint: str | None = None
    https_with_key: str | None = None
    wss_with_key: str | None = None
    beacon_with_key: str | None = None
    # Protocol-specific computed endpoints (e.g., avalanche C/X/P, TRON wallets, TON toncenter)
    extras: dict[str, str] | None = None


class CreateNodeResult(BaseModel):
    """Tool response to surface to the LLM."""

    node_id: str
    status: str
    network: str
    provider: str
    region: str
    configuration: NodeConfiguration
    endpoints: NodeEndpoints


def _get_api_key() -> str:
    """Retrieve the Chainstack Platform API key from the environment."""
    api_key = os.environ.get(CHAINSTACK_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"Set {CHAINSTACK_API_KEY_ENV} with a Chainstack Platform API key before creating nodes."
        )
    return api_key


async def _call_chainstack_api(payload: dict) -> dict:
    """Execute the HTTP POST to create a node and return the JSON response."""
    api_key = _get_api_key()
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{CHAINSTACK_API_BASE.rstrip('/')}/v1/nodes/",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
    if response.status_code >= 400:
        detail = response.text
        raise RuntimeError(
            f"Chainstack API responded with {response.status_code}: {detail}"
        )
    return response.json()


async def _chainstack_get(path: str, params: dict | None = None) -> dict:
    """Execute a GET request to the Chainstack API and return JSON."""
    api_key = _get_api_key()
    url = f"{CHAINSTACK_API_BASE.rstrip('/')}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
            params=params or {},
        )
    if response.status_code >= 400:
        detail = response.text
        raise RuntimeError(f"Chainstack API GET {path} failed with {response.status_code}: {detail}")
    return response.json()


def _canonical_public_network(public_network: str) -> str:
    """Normalize a provided public network string to Chainstack's canonical value."""
    key = (public_network or "").strip().lower()
    return NETWORK_ALIAS_TO_PUBLIC.get(key, key)


def _append_path(base_url: str, *segments: str) -> str:
    """Safely append path segments to a base URL.

    Preserves scheme/host, joins path with '/'. Ignores empty segments.
    """
    if not base_url:
        return base_url
    parsed = urlparse(base_url)
    existing = parsed.path.rstrip("/")
    add = "/".join(s.strip("/") for s in segments if s)
    new_path = "/".join(s for s in [existing, add] if s)
    return urlunparse(parsed._replace(path=f"/{new_path.lstrip('/')}"))


def _insert_key_before_path(base_url: str, auth_key: str) -> str:
    """Insert auth_key immediately after host, before existing path.

    Examples:
    - https://example.com -> https://example.com/{key}
    - https://example.com/evm -> https://example.com/{key}/evm
    - https://example.com/info -> https://example.com/{key}/info
    """
    if not base_url:
        return base_url
    parsed = urlparse(base_url)
    existing = parsed.path or ""
    new_path = f"/{auth_key}{existing if existing.startswith('/') else '/' + existing if existing else ''}"
    return urlunparse(parsed._replace(path=new_path))


def _build_host_key_url(base_url: str, auth_key: str, after_path: str) -> str:
    """Build URL as scheme://host/{auth_key}{after_path} ignoring base path."""
    parsed = urlparse(base_url)
    suffix = after_path if after_path.startswith('/') else f"/{after_path}"
    return urlunparse(parsed._replace(path=f"/{auth_key}{suffix}"))


async def _resolve_network_id(public_network: str) -> str:
    """Resolve a public network name (e.g. 'ethereum-mainnet') to a network ID (NW-...).

    Raises a ToolError with suggestions if no results are found.
    """
    normalized = _canonical_public_network(public_network)
    try:
        data = await _chainstack_get("/v1/networks/", params={"public_network": normalized})
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Failed to look up network '{normalized}': {exc}") from exc

    results = data.get("results") or []
    if not results:
        raise ToolError(
            "Unknown public network name. Try one of: ethereum-mainnet, ethereum-sepolia, ethereum-holesky."
        )
    return str(results[0]["id"])  # first match


async def _list_all_nodes() -> list[dict]:
    """Fetch all nodes across paginated responses."""
    page = 1
    results: list[dict] = []
    while True:
        data = await _chainstack_get("/v1/nodes/", params={"page": page})
        page_results = data.get("results") or []
        results.extend(page_results)
        if not data.get("next"):
            break
        page += 1
    return results


# -----------------------------------------------------------------------------
# Planned tool (comment-only spec): faucet_drip
# -----------------------------------------------------------------------------
# Intention: Action tool to request a testnet drip for a wallet address.
# Not implemented yet; backend faucet API will be private and called server-side.
#
# Rationale for tool (not resource): Tools represent side-effects and are not
# prefetched by clients, avoiding accidental drips that a dynamic resource read
# could trigger. Keep this as a tool to prevent unintended calls.
#
# Auth and scopes:
# - Server runs behind OAuth/OIDC (Keycloak). Require scope: faucet:drip
# - No end-user API key; server uses service credentials to call faucet backend
#
# Signature (planned):
# @mcp.tool(name="faucet_drip")
# async def faucet_drip(
#     network: Annotated[str, Field(description="Public network alias, e.g. 'ethereum-sepolia', 'polygon-amoy'")],
#     address: Annotated[str, Field(description="Recipient wallet address on the target network")],
#     amount: Annotated[str | None, Field(description="Optional human-readable amount; default chosen by server")]=None,
# ) -> dict:
#     """Request a testnet faucet transfer to `address` on `network`.
#
#     Guidance to LLMs: Only call when the user explicitly asks for faucet funds.
#     Do not poll or retry aggressively; respect rate limits. Prefer canonical
#     network aliases (e.g., 'ethereum-sepolia', 'polygon-amoy').
#     """
#     ...
#
# Expected response shape (example):
# {
#   "network": "ethereum-sepolia",
#   "address": "0xabc...",
#   "amount": "0.1 ETH",
#   "tx_hash": "0x...",
#   "explorer_url": "https://sepolia.etherscan.io/tx/0x...",
#   "status": "submitted|mined|queued",
#   "next_eligible_at": "2025-10-21T12:34:56Z"
# }
#
# Backend/implementation considerations:
# - Idempotency: generate an idempotency key by (network,address) and a time window
#   to prevent duplicate drips on retries. Return existing tx when duplicated.
# - Rate limiting: enforce per-address and per-user limits; add global quotas.
# - Allowlist networks: restrict to testnets only (e.g., ethereum-sepolia,
#   ethereum-holesky, polygon-amoy, bsc-testnet, avalanche-fuji, etc.).
# - Address validation: EVM checksum validation for EVM networks; backend performs
#   network-specific validation for non-EVM where applicable.
# - Observability: log network, address, amount, tx_hash, status; include request
#   ID and idempotency key; expose minimal details to clients.
# - Error taxonomy (map to ToolError messages):
#   * unauthorized (missing scope faucet:drip)
#   * network_unsupported (not in allowlist)
#   * invalid_address
#   * rate_limited (retry_after seconds)
#   * backend_unavailable / upstream_error
#   * insufficient_liquidity (temporarily out of funds)
#   * already_dripped_recently (idempotency/rate window)
# - Timeout: keep server call bounded (e.g., 10s). Return tx_hash quickly; let
#   clients poll explorer if needed instead of blocking for confirmations.
# - Security: do NOT expose backend endpoint publicly; invoke with server-side
#   service credentials only. Never echo secrets to clients.
#
# Example invocation (LLM intent):
# - "Send faucet on Polygon Amoy to 0xabc..." -> faucet_drip(network="polygon-amoy", address="0xabc...")
#
# Hex/JSON-RPC guidance reminder (EVM only): quantities must be 0x-prefixed hex
# (0x0 for zero). Do not hand-convert; use programmatic helpers (Python: hex(n);
# Node: '0x' + (n).toString(16); Bash: printf '0x%x' n) and verify via
# eth_getBlockByNumber that response.number matches.

def _pick_best_node_for_kind(nodes: list[dict], public_network: str, endpoint_kind: str) -> dict | None:
    """Choose the most suitable node for a requested endpoint kind.

    Preference order:
    - Same public_network (when we can infer from endpoints host naming)
    - chainstack_global/global1
    - Has required namespaces for debug/trace when endpoint_kind implies debug/tracing
    - Otherwise first running node with a matching endpoint present
    """
    def has_kind(n: dict) -> bool:
        d = n.get("details") or {}
        if endpoint_kind == "execution":
            return bool(d.get("https_endpoint"))
        if endpoint_kind == "beacon":
            return bool(d.get("beacon_endpoint"))
        if endpoint_kind == "avalanche_c":
            return bool(d.get("https_endpoint")) and (n.get("configuration", {}).get("client", "").lower() in {"avalanchego", "avax"})
        if endpoint_kind == "tron_jsonrpc":
            return "tron" in (d.get("https_endpoint") or "")
        if endpoint_kind == "hyperliquid_evm":
            return "hyperliquid" in (d.get("https_endpoint") or "")
        return bool(d.get("https_endpoint"))

    candidates = [n for n in nodes if n.get("status") == "running" and has_kind(n)]
    if not candidates:
        return None

    # Prefer chainstack_global/global1
    globals_first = [n for n in candidates if n.get("provider") == "chainstack_global" and n.get("region") == "global1"]
    if globals_first:
        return globals_first[0]
    return candidates[0]


async def _resolve_endpoint_url(
    *,
    ctx: Context,
    public_network: str,
    endpoint_kind: str,
) -> str:
    """Resolve an endpoint URL with auth key in path, using cache when possible."""
    session_id = ctx.session_id
    normalized = _canonical_public_network(public_network)

    # Cache hit
    if cached := _cache_get(session_id, normalized, endpoint_kind):
        return cached

    # Fallback: enumerate nodes and pick best
    raw_nodes = await _list_all_nodes()
    node = _pick_best_node_for_kind(raw_nodes, normalized, endpoint_kind)
    if not node:
        raise ToolError(f"No suitable node found for {normalized} ({endpoint_kind})")

    details = node.get("details", {}) or {}
    auth_key = details.get("auth_key")
    if not auth_key:
        raise ToolError("Node does not expose an auth key in details")

    url: str | None = None
    if endpoint_kind == "execution":
        url = _insert_key_before_path(details.get("https_endpoint"), auth_key)
    elif endpoint_kind == "beacon":
        url = _build_host_key_url(details.get("beacon_endpoint"), auth_key, "/")
    elif endpoint_kind == "avalanche_c":
        base_http = details.get("https_endpoint")
        url = _append_path(base_http, auth_key, "ext", "bc", "C", "rpc") if base_http else None
    elif endpoint_kind == "tron_jsonrpc":
        base = details.get("https_endpoint")
        url = _build_host_key_url(base, auth_key, "/jsonrpc") if base else None
    elif endpoint_kind == "hyperliquid_evm":
        base = details.get("https_endpoint")
        url = _build_host_key_url(base, auth_key, "/evm") if base else None
    else:
        raise ToolError(f"Unsupported endpoint_kind: {endpoint_kind}")

    if not url:
        raise ToolError("Failed to construct endpoint URL")

    _cache_set(session_id, normalized, endpoint_kind, url)
    return url

def _validate_global_archive(request: CreateNodeRequest) -> None:
    """Prevent known invalid combinations that trigger server errors downstream."""
    if request.provider == "chainstack_global" and request.configuration.archive:
        raise ValueError(
            "Global shared networks do not support archive=true. "
            "Set archive to false or choose a managed region."
        )


@mcp.tool(name="create_chainstack_node")
async def create_chainstack_node(
    name: Annotated[str, Field(min_length=3)],
    public_network: str,
    node_type: Literal["shared"] = "shared",
    archive: bool = False,
    role: Literal["peer", "validator"] = "peer",
) -> CreateNodeResult:
    """Provision a Chainstack node with simple, flat parameters.

    Guidance: Prefer Chainstack endpoints for on‑chain data. After creation,
    use the returned `https_with_key` (EVM), `beacon_with_key` (beacon), or
    protocol extras (Avalanche C/X/P, Tron JSON‑RPC, Hyperliquid EVM/info).
    EVM only: Don't hand‑convert decimals ↔ hex; use programmatic helpers
    (Python: hex(n); Node: '0x' + (n).toString(16); Bash: printf '0x%x' n)
    and verify via eth_getBlockByNumber that response.number matches.

    - Resolves `public_network` (e.g., 'ethereum-mainnet', 'sepolia') to a network ID
    - For shared nodes, uses provider=chainstack_global and region=global1
    - For dedicated nodes, requires explicit `provider` and `region`
    """
    # Resolve network ID from public network name
    network_id = await _resolve_network_id(public_network)

    # Shared nodes are enforced; provider/region fixed
    provider_final = "chainstack_global"
    region_final = "global1"
    if archive:
        raise ToolError(
            "Global shared networks do not support archive=true. Set archive to false."
        )

    payload = {
        "name": name,
        "network": network_id,
        "provider": provider_final,
        "region": region_final,
        "role": role,
        "type": node_type,
        "configuration": NodeConfiguration(archive=archive).model_dump(
            exclude_none=True
        ),
    }

    try:
        data = await _call_chainstack_api(payload)
    except Exception as exc:  # noqa: BLE001 - surface HTTP+runtime errors cleanly
        raise ToolError(f"Failed to create node: {exc}") from exc

    details = data.get("details", {}) or {}
    auth_key = details.get("auth_key")

    # Build key-appended endpoints without exposing basic-auth credentials
    https_with_key = (
        _insert_key_before_path(details.get("https_endpoint"), auth_key)
        if details.get("https_endpoint") and auth_key
        else None
    )
    wss_with_key = (
        _insert_key_before_path(details.get("wss_endpoint"), auth_key)
        if details.get("wss_endpoint") and auth_key
        else None
    )
    beacon_with_key = (
        _append_path(details.get("beacon_endpoint"), auth_key)
        if details.get("beacon_endpoint") and auth_key
        else None
    )

    # Compute protocol-specific extras
    extras: dict[str, str] = {}
    # TON toncenter v2/v3 (insert key before /api path)
    if details.get("toncenter_api_v2") and auth_key:
        extras["toncenter_api_v2"] = _build_host_key_url(
            details["toncenter_api_v2"], auth_key, "/api/v2"
        )
    if details.get("toncenter_api_v3") and auth_key:
        extras["toncenter_api_v3"] = _build_host_key_url(
            details["toncenter_api_v3"], auth_key, "/api/v3"
        )
    # TRON JSON-RPC and wallet endpoints (insert key immediately after host)
    if "tron" in (details.get("https_endpoint") or "") and auth_key:
        base = details.get("https_endpoint")
        extras["tron_jsonrpc"] = _build_host_key_url(base, auth_key, "/jsonrpc")
        extras["tron_http_wallet"] = _build_host_key_url(base, auth_key, "/wallet")
        extras["tron_http_walletsolidity"] = _build_host_key_url(
            base, auth_key, "/walletsolidity"
        )
    # Avalanche C/X/P chain paths
    if (
        (details.get("https_endpoint") or "").endswith(".p2pify.com")
        and auth_key
        and (configuration.client or "").lower() in {"avalanchego", "avax"}
    ):
        base_http = details.get("https_endpoint")
        base_ws = details.get("wss_endpoint")
        extras["avalanche_c_http"] = _append_path(base_http, auth_key, "ext", "bc", "C", "rpc")
        if base_ws:
            extras["avalanche_c_ws"] = _append_path(base_ws, auth_key, "ext", "bc", "C", "ws")
        extras["avalanche_x_http"] = _append_path(base_http, auth_key, "ext", "bc", "X")
        if base_ws:
            extras["avalanche_x_ws"] = _append_path(base_ws, auth_key, "ext", "bc", "X", "events")
        extras["avalanche_p_http"] = _append_path(base_http, auth_key, "ext", "P")
    # Hyperliquid EVM/info path style
    if "hyperliquid" in (details.get("https_endpoint") or "") and auth_key:
        base = details.get("https_endpoint")
        # Required form: https://host/{key}/evm and https://host/{key}/info
        extras["hyperliquid_jsonrpc"] = _build_host_key_url(base, auth_key, "/evm")
        extras["hyperliquid_info"] = _build_host_key_url(
            details.get("info_endpoint") or base, auth_key, "/info"
        )

    return CreateNodeResult(
        node_id=data["id"],
        status=data.get("status", "unknown"),
        network=data["network"],
        provider=data["provider"],
        region=data["region"],
        configuration=NodeConfiguration(archive=archive),
        endpoints=NodeEndpoints(
            https_endpoint=details.get("https_endpoint"),
            wss_endpoint=details.get("wss_endpoint"),
            beacon_endpoint=details.get("beacon_endpoint"),
            https_with_key=https_with_key,
            wss_with_key=wss_with_key,
            beacon_with_key=beacon_with_key,
            extras=extras or None,
        ),
    )


@mcp.tool(name="list_chainstack_nodes")
async def list_chainstack_nodes(ctx: Context | None = None) -> list[CreateNodeResult]:
    """List all nodes in the organization with sanitized connection info.

    Guidance: Prefer Chainstack endpoints for on‑chain data. Map endpoint_kind
    to URLs:
    - execution → `https_with_key` (EVM chains)
    - beacon → `beacon_with_key` (Ethereum/Gnosis consensus)
    - avalanche_c → `extras.avalanche_c_http`
    - tron_jsonrpc → `extras.tron_jsonrpc`
    - hyperliquid_evm → `extras.hyperliquid_jsonrpc`
    - solana_jsonrpc → `https_with_key`
    EVM only: JSON‑RPC quantities MUST be 0x‑prefixed hex (0x0 for zero). If
    needed, don't hand‑convert decimals ↔ hex; use programmatic helpers
    (Python: hex(n); Node: '0x' + (n).toString(16); Bash: printf '0x%x' n),
    then verify via eth_getBlockByNumber that response.number matches. If
    “method not found”, verify endpoint_kind and hex parameter encoding.

    Returns a list of nodes including ID, status, network ID, provider/region,
    and safe endpoints with the auth key appended to the URL path.
    """
    try:
        raw_nodes = await _list_all_nodes()
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Failed to list nodes: {exc}") from exc

    results: list[CreateNodeResult] = []
    for n in raw_nodes:
        details = n.get("details", {}) or {}
        auth_key = details.get("auth_key")

        https_with_key = (
            _insert_key_before_path(details.get("https_endpoint"), auth_key)
            if details.get("https_endpoint") and auth_key
            else None
        )
        wss_with_key = (
            _insert_key_before_path(details.get("wss_endpoint"), auth_key)
            if details.get("wss_endpoint") and auth_key
            else None
        )
        beacon_with_key = (
            _append_path(details.get("beacon_endpoint"), auth_key)
            if details.get("beacon_endpoint") and auth_key
            else None
        )

        configuration = NodeConfiguration(**(n.get("configuration", {}) or {}))

        # Compute protocol-specific extras
        extras: dict[str, str] = {}
        if details.get("toncenter_api_v2") and auth_key:
            extras["toncenter_api_v2"] = _build_host_key_url(
                details["toncenter_api_v2"], auth_key, "/api/v2"
            )
        if details.get("toncenter_api_v3") and auth_key:
            extras["toncenter_api_v3"] = _build_host_key_url(
                details["toncenter_api_v3"], auth_key, "/api/v3"
            )
        if "tron" in (details.get("https_endpoint") or "") and auth_key:
            base = details.get("https_endpoint")
            extras["tron_jsonrpc"] = _build_host_key_url(base, auth_key, "/jsonrpc")
            extras["tron_http_wallet"] = _build_host_key_url(base, auth_key, "/wallet")
            extras["tron_http_walletsolidity"] = _build_host_key_url(
                base, auth_key, "/walletsolidity"
            )
        if (
            (details.get("https_endpoint") or "").endswith(".p2pify.com")
            and auth_key
            and (configuration.client or "").lower() in {"avalanchego", "avax"}
        ):
            base_http = details.get("https_endpoint")
            base_ws = details.get("wss_endpoint")
            extras["avalanche_c_http"] = _append_path(base_http, auth_key, "ext", "bc", "C", "rpc")
            if base_ws:
                extras["avalanche_c_ws"] = _append_path(base_ws, auth_key, "ext", "bc", "C", "ws")
            extras["avalanche_x_http"] = _append_path(base_http, auth_key, "ext", "bc", "X")
            if base_ws:
                extras["avalanche_x_ws"] = _append_path(base_ws, auth_key, "ext", "bc", "X", "events")
            extras["avalanche_p_http"] = _append_path(base_http, auth_key, "ext", "P")
        if "hyperliquid" in (details.get("https_endpoint") or "") and auth_key:
            base = details.get("https_endpoint")
            # Required form: https://host/{key}/evm and https://host/{key}/info
            extras["hyperliquid_jsonrpc"] = _build_host_key_url(base, auth_key, "/evm")
            extras["hyperliquid_info"] = _build_host_key_url(
                details.get("info_endpoint") or base, auth_key, "/info"
            )

        results.append(
            CreateNodeResult(
                node_id=str(n.get("id")),
                status=str(n.get("status", "unknown")),
                network=str(n.get("network", "unknown")),
                provider=str(n.get("provider", "unknown")),
                region=str(n.get("region", "unknown")),
                configuration=configuration,
                endpoints=NodeEndpoints(
                    https_endpoint=details.get("https_endpoint"),
                    wss_endpoint=details.get("wss_endpoint"),
                    beacon_endpoint=details.get("beacon_endpoint"),
                    https_with_key=https_with_key,
                    wss_with_key=wss_with_key,
                    beacon_with_key=beacon_with_key,
                    extras=extras or None,
                ),
            )
        )

    # Warm per-session cache with first candidate per network/kind for speed
    if ctx is not None:
        session_id = ctx.session_id
        for r in results:
            pub_guess = str(r.network)  # network ID; cache keyed by network id works too
            if r.endpoints.https_with_key:
                _cache_set(session_id, pub_guess, "execution", r.endpoints.https_with_key)
            if r.endpoints.beacon_with_key:
                _cache_set(session_id, pub_guess, "beacon", r.endpoints.beacon_with_key)
            if r.endpoints.extras:
                if r.endpoints.extras.get("avalanche_c_http"):
                    _cache_set(session_id, pub_guess, "avalanche_c", r.endpoints.extras["avalanche_c_http"])  # type: ignore[index]
                if r.endpoints.extras.get("tron_jsonrpc"):
                    _cache_set(session_id, pub_guess, "tron_jsonrpc", r.endpoints.extras["tron_jsonrpc"])  # type: ignore[index]
                if r.endpoints.extras.get("hyperliquid_jsonrpc"):
                    _cache_set(session_id, pub_guess, "hyperliquid_evm", r.endpoints.extras["hyperliquid_jsonrpc"])  # type: ignore[index]

    return results


def main() -> None:
    """Entry point for running the Streamable HTTP server."""
    host = os.environ.get("CHAINSTACK_MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAINSTACK_MCP_PORT", "8000"))
    path = os.environ.get("CHAINSTACK_MCP_PATH", "/mcp")
    mcp.run_streamable_http(host=host, port=port, path=path)


if __name__ == "__main__":
    main()
