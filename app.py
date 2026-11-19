import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
import re
from urllib.parse import unquote, urljoin, urlparse

import anthropic
import requests
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(page_title="API Chat Assistant", page_icon="💬")

HEADERS = {"User-Agent": "Mozilla/5.0 (DocAutoClient)"}

COMMON_SPEC_PATHS = [
    "/openapi.json", "/openapi.yaml", "/swagger.json", "/swagger.yaml",
    "/api-docs", "/v1/openapi.json", "/v3/api-docs", "/docs/openapi.json",
]

PROVIDERS = {
    "Claude (Anthropic)": {"type": "anthropic", "base_url": None},
    "OpenAI": {"type": "openai", "base_url": None},
    "Grok (xAI)": {"type": "openai", "base_url": "https://api.x.ai/v1"},
    "DeepSeek": {"type": "openai", "base_url": "https://api.deepseek.com"},
    "Gemini (Google)": {
        "type": "openai",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "Groq": {"type": "openai", "base_url": "https://api.groq.com/openai/v1"},
    "OpenRouter": {"type": "openai", "base_url": "https://openrouter.ai/api/v1"},
    "Custom (OpenAI-compatible)": {"type": "openai", "base_url": ""},
}

# ---------------------------------------------------------------- tool definitions
TOOL_DEFS = [
    {
        "name": "search_docs",
        "description": (
            "Search the web to find the official API documentation page of a website or "
            "service when the user gave only its name. Returns a list of {title, url}. "
            "Try queries like '<name> API documentation' or '<name> REST API reference'. "
            "Prefer the official domain of the service over blogs or third-party sites."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "read_url",
        "description": (
            "Download a web page (for example API documentation) and return its text, "
            "OpenAPI/Swagger links found on it, and the page's links so you can follow "
            "them. Call it again with other URLs from the page to read deeper."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "load_openapi",
        "description": (
            "Try to find and load an OpenAPI/Swagger spec for a documentation URL. "
            "Returns base_url, auth info and the endpoint list, or found=false."
        ),
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "set_credential",
        "description": (
            "Store a secret the user typed in chat (API key, token, account id) under a "
            "short name. For HTTP Basic auth store 'username:password'. "
            "Never repeat the secret in your replies."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "value": {"type": "string"}},
            "required": ["name", "value"],
        },
    },
    {
        "name": "call_api",
        "description": (
            "Send a real HTTP request to an API. url can be a full URL or a path (then "
            "the loaded base_url is used). Some APIs (for example Twilio) expect "
            "form-encoded bodies: set body_format to 'form'. auth.credential is the NAME "
            "of a stored credential; auth.type is none, bearer, basic (credential stored "
            "as 'user:password'), header or query; auth.name is the header or query "
            "parameter name when needed. params, body and headers must be JSON objects written "
            "as strings, for example '{\"To\": \"+15551234567\"}'. Anywhere in url, headers, "
            "params or body you may "
            "write {{cred:NAME}} and the app inserts the stored secret itself. To keep a "
            "value from the response (for example an OAuth access_token) as a new stored "
            "credential without seeing it, set save_as."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "url": {"type": "string"},
                "params": {"type": "string"},
                "body": {"type": "string"},
                "body_format": {"type": "string", "enum": ["json", "form"]},
                "headers": {"type": "string"},
                "save_as": {
                    "type": "object",
                    "properties": {
                        "credential": {"type": "string"},
                        "json_field": {"type": "string"},
                    },
                },
                "auth": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "credential": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            },
            "required": ["method", "url"],
        },
    },
    {
        "name": "setup_client",
        "description": (
            "Save the REST client setup you worked out from the documentation: base URL, "
            "authorization method and the endpoints you found. It is remembered for the "
            "rest of the chat. auth.credential is a short name you choose for the secret "
            "(for example 'api_key')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "base_url": {"type": "string"},
                "auth": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "credential": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
                "endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "path": {"type": "string"},
                            "summary": {"type": "string"},
                            "params": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["base_url"],
        },
    },
]
ANTHROPIC_TOOLS = [
    {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
    for t in TOOL_DEFS
]
OPENAI_TOOLS = [{"type": "function", "function": t} for t in TOOL_DEFS]


# ---------------------------------------------------------------- OpenAPI helpers
def try_load_spec(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        try:
            data = json.loads(r.text)
        except Exception:
            try:
                data = yaml.safe_load(r.text)
            except Exception:
                return None
        if isinstance(data, dict) and ("openapi" in data or "swagger" in data):
            return data
    except Exception:
        return None
    return None


def find_openapi(doc_url):
    spec = try_load_spec(doc_url)
    if spec:
        return spec
    try:
        html = requests.get(doc_url, headers=HEADERS, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all(["a", "link", "script"]):
            ref = tag.get("href") or tag.get("src") or ""
            if re.search(r"(openapi|swagger).*\.(json|ya?ml)$", ref, re.I):
                spec = try_load_spec(urljoin(doc_url, ref))
                if spec:
                    return spec
    except Exception:
        pass
    parsed = urlparse(doc_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    for path in COMMON_SPEC_PATHS:
        spec = try_load_spec(base + path)
        if spec:
            return spec
    return None


def spec_to_config(spec, doc_url):
    base_url = ""
    if spec.get("servers"):
        base_url = spec["servers"][0].get("url", "")
    elif "host" in spec:
        scheme = (spec.get("schemes") or ["https"])[0]
        base_url = f"{scheme}://{spec['host']}{spec.get('basePath', '')}"
    if base_url and not base_url.startswith("http"):
        base_url = urljoin(doc_url, base_url)

    auth = {"type": "bearer", "name": "Authorization"}
    schemes = (spec.get("components", {}).get("securitySchemes")
               or spec.get("securityDefinitions") or {})
    for s in schemes.values():
        if s.get("type") == "apiKey":
            where = "query" if s.get("in") == "query" else "header"
            auth = {"type": where, "name": s.get("name", "X-API-Key")}
            break
        if s.get("type") == "http" and s.get("scheme") == "basic":
            auth = {"type": "basic"}
            break

    endpoints = []
    for path, methods in spec.get("paths", {}).items():
        if not isinstance(methods, dict):
            continue
        for method, info in methods.items():
            if method.lower() not in ("get", "post", "put", "patch", "delete"):
                continue
            if not isinstance(info, dict):
                continue
            params = [p.get("name") for p in info.get("parameters", [])
                      if isinstance(p, dict) and p.get("name")]
            endpoints.append({
                "method": method.upper(), "path": path,
                "summary": info.get("summary", ""), "params": params,
            })
    return {"base_url": base_url, "auth": auth, "endpoints": endpoints}


# ---------------------------------------------------------------- tool implementations
def tool_read_url(url):
    r = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    spec_links, links = [], []
    host = urlparse(url).netloc
    for tag in soup.find_all(["a", "link", "script"]):
        ref = tag.get("href") or tag.get("src") or ""
        if not ref or ref.startswith("#"):
            continue
        full = urljoin(url, ref)
        if re.search(r"(openapi|swagger).*\.(json|ya?ml)$", ref, re.I):
            spec_links.append(full)
        elif tag.name == "a" and urlparse(full).netloc == host and len(links) < 60:
            label = tag.get_text(" ", strip=True)[:60]
            if label:
                links.append(f"{label} -> {full}")
    for t in soup(["script", "style"]):
        t.decompose()
    text = re.sub(r"\n\s*\n+", "\n", soup.get_text("\n"))
    return {
        "status": r.status_code,
        "text": text[:15000],
        "openapi_links": spec_links[:10],
        "page_links": links,
    }


def ddg_search(query, n=8):
    r = requests.post("https://html.duckduckgo.com/html/", data={"q": query},
                      headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        m = re.search(r"uddg=([^&]+)", href)
        if m:
            href = unquote(m.group(1))
        if href.startswith("//"):
            href = "https:" + href
        if href.startswith("http"):
            results.append({"title": a.get_text(" ", strip=True)[:100], "url": href})
        if len(results) >= n:
            break
    return results


def bing_search(query, n=8):
    r = requests.get("https://www.bing.com/search", params={"q": query},
                     headers=HEADERS, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    results = []
    for a in soup.select("li.b_algo h2 a"):
        href = a.get("href", "")
        if href.startswith("http"):
            results.append({"title": a.get_text(" ", strip=True)[:100], "url": href})
        if len(results) >= n:
            break
    return results


def _probe(u):
    try:
        r = requests.get(u, headers=HEADERS, timeout=4, allow_redirects=True)
        return r.url if r.status_code == 200 else None
    except Exception:
        return None


def guess_doc_urls(query):
    """Last fallback: probe common documentation addresses in parallel."""
    words = re.sub(r"(?i)\b(api|docs?|documentation|rest|reference|developer)\b", " ", query)
    slug = re.sub(r"[^a-z0-9]", "", words.lower())
    if not slug:
        return []
    candidates = [
        f"https://docs.{slug}.com", f"https://developers.{slug}.com",
        f"https://developer.{slug}.com", f"https://www.{slug}.com/docs",
        f"https://{slug}.com/docs", f"https://api.{slug}.com",
        f"https://{slug}.com/developers", f"https://{slug}.dev/docs",
    ]
    with ThreadPoolExecutor(max_workers=8) as ex:
        hits = [h for h in ex.map(_probe, candidates) if h]
    return [{"title": "guessed address", "url": h} for h in hits]


def tool_search_docs(query):
    results = []
    for fn in (ddg_search, bing_search):
        try:
            results = fn(query)
        except Exception:
            results = []
        if results:
            break
    if not results:
        results = guess_doc_urls(query)
    if not results:
        return {
            "found": False,
            "message": "No results. Do not search again more than once; "
                       "ask the user for the documentation link.",
        }
    return {"found": True, "results": results}


def tool_load_openapi(url):
    spec = find_openapi(url)
    if not spec:
        return {"found": False}
    cfg = spec_to_config(spec, url)
    st.session_state.config = cfg
    return {
        "found": True, "base_url": cfg["base_url"], "auth": cfg["auth"],
        "total_endpoints": len(cfg["endpoints"]), "endpoints": cfg["endpoints"][:150],
    }


def tool_set_credential(name, value):
    st.session_state.credentials[name] = value.strip()
    return {"saved": name}


def needs_confirmation(args):
    """Code-level safety: a write call only runs after the user replied in a later chat turn."""
    if args["method"].upper() in ("GET", "HEAD", "OPTIONS"):
        return False
    sig = hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()
    turn = len([m for m in st.session_state.history if m["role"] == "user"])
    pending = st.session_state.get("pending")
    if pending and pending["sig"] == sig and turn > pending["turn"]:
        st.session_state.pending = None
        return False
    st.session_state.pending = {"sig": sig, "turn": turn}
    return True


def as_obj(v):
    """params/body/headers arrive as JSON strings (or dicts); always return a dict."""
    if isinstance(v, dict):
        return dict(v)
    if isinstance(v, str) and v.strip():
        try:
            parsed = json.loads(v)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def get_field(data, path):
    for part in path.split("."):
        if isinstance(data, dict) and part in data:
            data = data[part]
        else:
            return None
    return data


def subst(obj):
    """Replace {{cred:NAME}} with the stored secret. Done in code, so the AI never sees it."""
    if isinstance(obj, str):
        def rep(m):
            name = m.group(1).strip()
            if name not in st.session_state.credentials:
                raise KeyError(f"credential '{name}' is not stored yet. Ask the user for it.")
            return st.session_state.credentials[name]
        return re.sub(r"\{\{cred:([^}]+)\}\}", rep, obj)
    if isinstance(obj, dict):
        return {k: subst(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [subst(v) for v in obj]
    return obj


def tool_setup_client(base_url, auth=None, endpoints=None, notes=""):
    cfg = dict(st.session_state.get("config") or {})
    cfg["base_url"] = base_url
    if auth:
        cfg["auth"] = auth
    if endpoints:
        cfg["endpoints"] = endpoints
    if notes:
        cfg["notes"] = notes
    cfg.setdefault("endpoints", [])
    st.session_state.config = cfg
    return {"saved": True, "base_url": base_url, "endpoints": len(cfg["endpoints"])}


def tool_call_api(args):
    method = args["method"].upper()
    cfg = st.session_state.get("config") or {}
    raw_url = args["url"]

    # Token requests that only save a token are harmless; every other write needs the user's yes.
    is_token_call = bool(args.get("save_as")) and bool(re.search(r"oauth|token|auth", raw_url, re.I))
    if not is_token_call and needs_confirmation(args):
        return {
            "status": "NOT_SENT",
            "message": (
                "Not sent yet. Show the user exactly what will be sent (method, url, "
                "params, body) in simple words and ask them to reply yes in chat. After "
                "they reply, call call_api again with exactly the same arguments."
            ),
        }

    url = subst(raw_url)
    params = subst(as_obj(args.get("params")))
    headers = subst(as_obj(args.get("headers")))
    body = subst(as_obj(args.get("body")))
    if not url.startswith("http"):
        url = cfg.get("base_url", "").rstrip("/") + "/" + url.lstrip("/")
    for k in list(params.keys()):
        if "{" + k + "}" in url:
            url = url.replace("{" + k + "}", str(params.pop(k)))

    auth = args.get("auth")
    if auth is None:
        cfg_auth = cfg.get("auth") or {}
        auth = cfg_auth if cfg_auth.get("credential") else {}
    atype = auth.get("type", "none")
    basic = None
    if atype != "none":
        cred = st.session_state.credentials.get(auth.get("credential", ""))
        if cred is None:
            return {"error": "credential not stored yet. Ask the user for it."}
        if atype == "bearer":
            headers["Authorization"] = f"Bearer {cred}"
        elif atype == "header":
            headers[auth.get("name", "X-API-Key")] = cred
        elif atype == "query":
            params[auth.get("name", "api_key")] = cred
        elif atype == "basic":
            user, _, pw = cred.partition(":")
            basic = (user, pw)

    kwargs = {"headers": headers, "params": params, "auth": basic, "timeout": 30}
    if body:
        if args.get("body_format") == "form":
            kwargs["data"] = body
        else:
            kwargs["json"] = body
    resp = requests.request(method, url, **kwargs)
    try:
        data = resp.json()
    except Exception:
        data = resp.text
    result = {"status": resp.status_code, "body": data}

    save_as = args.get("save_as")
    if save_as and isinstance(data, dict):
        val = get_field(data, save_as.get("json_field", "access_token"))
        if val is not None:
            name = save_as.get("credential", "token")
            st.session_state.credentials[name] = str(val)
            result["saved_credential"] = name
    return result


def dispatch(name, args):
    try:
        if name == "search_docs":
            out = tool_search_docs(args["query"])
        elif name == "read_url":
            out = tool_read_url(args["url"])
        elif name == "load_openapi":
            out = tool_load_openapi(args["url"])
        elif name == "set_credential":
            out = tool_set_credential(args["name"], args["value"])
        elif name == "setup_client":
            out = tool_setup_client(
                args["base_url"], args.get("auth"), args.get("endpoints"), args.get("notes", ""))
        elif name == "call_api":
            out = tool_call_api(args)
        else:
            out = {"error": f"unknown tool {name}"}
    except Exception as e:
        out = {"error": str(e)}
    text = json.dumps(out, ensure_ascii=False, default=str)
    for secret in st.session_state.credentials.values():
        if len(secret) >= 6:
            text = text.replace(secret, "[hidden]")
    return text[:20000]


def show_tool(name, args):
    if name == "set_credential":
        label, shown = "🔐 saving a secret", {"name": args.get("name"), "value": "••••••"}
    elif name == "search_docs":
        label, shown = f"🌐 searching: {args.get('query')}", args
    elif name == "setup_client":
        label, shown = "🧩 REST client setup", args
    elif name in ("read_url", "load_openapi"):
        label, shown = f"🔎 {name}: {args.get('url')}", args
    else:
        label, shown = f"📡 {args.get('method')} {args.get('url')}", args
    st.caption(label)


# ---------------------------------------------------------------- AI layer
def assistant_msg(m):
    """Rebuild the assistant message, keeping extra fields (Gemini thought signatures)."""
    calls = []
    for tc in m.tool_calls:
        d = {
            "id": tc.id, "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        extra = (getattr(tc, "model_extra", None) or {}).get("extra_content")
        if extra:
            d["extra_content"] = extra
        calls.append(d)
    return {"role": "assistant", "content": m.content or "", "tool_calls": calls}


def ask_ai(provider, ai_key, model, base_url, system, history, use_tools=True):
    if provider["type"] == "anthropic":
        client = anthropic.Anthropic(api_key=ai_key, timeout=60, max_retries=1)
        msgs = list(history)
        for _ in range(8):
            kwargs = dict(model=model, max_tokens=2500, system=system, messages=msgs)
            if use_tools:
                kwargs["tools"] = ANTHROPIC_TOOLS
            r = client.messages.create(**kwargs)
            blocks = []
            for b in r.content:
                if b.type == "text":
                    blocks.append({"type": "text", "text": b.text})
                elif b.type == "tool_use":
                    blocks.append({"type": "tool_use", "id": b.id,
                                   "name": b.name, "input": b.input})
            msgs.append({"role": "assistant", "content": blocks})
            if r.stop_reason != "tool_use":
                return "".join(b["text"] for b in blocks if b["type"] == "text")
            results = []
            for b in r.content:
                if b.type == "tool_use":
                    show_tool(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": dispatch(b.name, b.input)})
            msgs.append({"role": "user", "content": results})
        return "(too many steps)"

    client = OpenAI(api_key=ai_key, base_url=base_url or None, timeout=60, max_retries=1)
    msgs = [{"role": "system", "content": system}] + list(history)
    for _ in range(8):
        kwargs = dict(model=model, messages=msgs)
        if use_tools:
            kwargs["tools"] = OPENAI_TOOLS
        r = client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append(assistant_msg(m))
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            show_tool(tc.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": dispatch(tc.function.name, args)})
    return "(too many steps)"


def build_system():
    s = (
        "You are a helpful, friendly general-purpose AI assistant, just like Claude. The "
        "user chats with you normally and you can help with anything (questions, writing, "
        "code, explanations), always in the language the user writes in (Bengali if they "
        "write Bengali). On top of that you have tools that let you work with REST APIs "
        "in the backend. "
        "When the user gives an API documentation link or just the name of a website or "
        "service, or asks to connect to some API "
        "(for example a phone/telephony provider for outbound calls), do the technical "
        "work yourself and do not ask the user to do any technical step: "
        "1) If the user gave only a website or service name (no link), call search_docs "
        "yourself to find its official API documentation page (queries like "
        "'<name> API documentation' and '<name> REST API reference'; prefer the official "
        "domain over blogs) and tell the user in one short line which link you picked. "
        "Then read_url the link, follow the relevant documentation links, and try "
        "load_openapi, until you know the base URL, the authorization method, and the "
        "exact endpoint, parameters and body format for what the user wants. "
        "2) Call setup_client to save the REST client setup (base URL, auth type, header "
        "or query name, a short credential name, the endpoints you found). "
        "3) Tell the user in one or two simple sentences that the client is ready, and "
        "ask ONLY for what authorization needs: the API key, or the client id (and client "
        "secret, account SID, etc. if the docs require them). Nothing else yet. "
        "4) When the user gives them, store each with set_credential and never repeat a "
        "secret. For OAuth client credentials, call the token endpoint with call_api, use "
        "{{cred:NAME}} placeholders for the secrets, and use save_as to keep the access "
        "token without seeing it. "
        "5) Test the client with a harmless read-only request (GET, for example account "
        "info or a list endpoint) and tell the user simply whether authorization worked. "
        "If it failed, explain what the error means and what to fix. "
        "Whenever a secret is needed in url, headers, params or body, write "
        "{{cred:NAME}} so the app inserts it and the secret never passes through you. "
        "Never invent endpoints or parameters: only use what you actually read. Real "
        "actions (POST, PUT, PATCH, DELETE, for example placing a phone call) cost money "
        "or affect real people, so show exactly what will be sent and wait for the user's "
        "yes in chat. Keep answers short and simple."
    )
    creds = list(st.session_state.credentials.keys())
    s += f"\nStored credential names: {creds if creds else 'none'}."
    cfg = st.session_state.get("config")
    if cfg:
        s += "\nCurrently set-up REST client: " + json.dumps(cfg, ensure_ascii=False)[:12000]
    return s


# ---------------------------------------------------------------- UI
st.session_state.setdefault("history", [])
st.session_state.setdefault("credentials", {})

st.title("💬 API Chat Assistant")
st.caption("সাধারণ চ্যাটের মতো যেকোনো কিছু জিজ্ঞেস করুন। শুধু ওয়েবসাইট বা সার্ভিসের নাম বললেই হবে, documentation আমি নিজে খুঁজে REST client সেটআপ করব, আর শুধু আপনার API key বা client id চাইব।")

with st.sidebar:
    st.subheader("AI সেটিংস")
    names = list(PROVIDERS.keys())
    pname = st.selectbox("Provider", names, index=names.index("Gemini (Google)"))
    provider = PROVIDERS[pname]
    base_url = provider["base_url"]
    if pname.startswith("Custom"):
        base_url = st.text_input("Base URL", placeholder="https://.../v1")
    model = st.text_input(
        "Model name",
        value="gemini-flash-latest" if pname.startswith("Gemini") else "",
        key=f"model_{pname}",
    ).strip()
    ai_key = st.text_input("AI API key", type="password", key=f"aikey_{pname}").strip()
    st.caption(f"এখন বাছা provider: {pname}")

    if st.button("Test this AI key"):
        if not ai_key or not model:
            st.warning("Model name আর AI key দিন")
        else:
            try:
                reply = ask_ai(provider, ai_key, model, base_url, "Reply with one word.",
                               [{"role": "user", "content": "Say hi"}], use_tools=False)
                st.success(f"Key works: {reply[:80]}")
            except Exception as e:
                st.error(f"Failed: {e}")

    with st.expander("🔑 Secret key দেওয়ার নিরাপদ জায়গা (ঐচ্ছিক)"):
        st.caption("চ্যাটে key লিখলে সেটা AI কোম্পানির কাছেও যায়। এখানে দিলে যায় না।")
        c_name = st.text_input("নাম (যেমন twilio)")
        c_val = st.text_input("Key বা token", type="password")
        if st.button("Save secret") and c_name and c_val:
            st.session_state.credentials[c_name.strip()] = c_val.strip()
            st.success("সেভ হয়েছে")
        if st.session_state.credentials:
            st.write("Saved:", ", ".join(st.session_state.credentials.keys()))

    if st.button("নতুন চ্যাট শুরু করুন"):
        st.session_state.history = []
        st.session_state.pop("config", None)
        st.session_state.pop("pending", None)
        st.rerun()

if not st.session_state.history:
    st.info(
        "যেমন লিখতে পারেন: \"Twilio দিয়ে REST এ outbound phone call করতে চাই, সেটআপ করে দাও\""
    )

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_text = st.chat_input("এখানে লিখুন...")
if user_text:
    if not ai_key or not model:
        st.warning("আগে সাইডবারে Model name আর AI API key দিন।")
        st.stop()
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                final = ask_ai(provider, ai_key, model, base_url, build_system(),
                               st.session_state.history)
            except Exception as e:
                final = f"Error: {e}"
        final = final or "(AI থেকে কোনো উত্তর আসেনি। Model name আর key ঠিক আছে কিনা দেখুন।)"
        st.write(final)
    st.session_state.history.append({"role": "assistant", "content": final})
