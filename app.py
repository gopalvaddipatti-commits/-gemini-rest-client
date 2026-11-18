import hashlib
import json
import re
from urllib.parse import urljoin, urlparse

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
            "of a stored credential; auth.type is none, bearer, basic, header or query; "
            "auth.name is the header or query parameter name when needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "method": {"type": "string"},
                "url": {"type": "string"},
                "params": {"type": "object"},
                "body": {"type": "object"},
                "body_format": {"type": "string", "enum": ["json", "form"]},
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


def tool_call_api(args):
    method = args["method"].upper()
    url = args["url"]
    params = dict(args.get("params") or {})
    if not url.startswith("http"):
        base = (st.session_state.get("config") or {}).get("base_url", "")
        url = base.rstrip("/") + "/" + url.lstrip("/")
    for k in list(params.keys()):
        if "{" + k + "}" in url:
            url = url.replace("{" + k + "}", str(params.pop(k)))

    if needs_confirmation(args):
        return {
            "status": "NOT_SENT",
            "message": (
                "Not sent yet. Show the user exactly what will be sent (method, url, "
                "params, body) in simple words and ask them to reply yes in chat. After "
                "they reply, call call_api again with exactly the same arguments."
            ),
        }

    headers, basic = {}, None
    auth = args.get("auth") or {}
    atype = auth.get("type", "none")
    if atype != "none":
        cred = st.session_state.credentials.get(auth.get("credential", ""))
        if cred is None:
            return {"error": "credential not stored. Ask the user for it."}
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
    body = args.get("body")
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
    return {"status": resp.status_code, "body": data}


def dispatch(name, args):
    try:
        if name == "read_url":
            out = tool_read_url(args["url"])
        elif name == "load_openapi":
            out = tool_load_openapi(args["url"])
        elif name == "set_credential":
            out = tool_set_credential(args["name"], args["value"])
        elif name == "call_api":
            out = tool_call_api(args)
        else:
            out = {"error": f"unknown tool {name}"}
    except Exception as e:
        out = {"error": str(e)}
    return json.dumps(out, ensure_ascii=False, default=str)[:20000]


def show_tool(name, args):
    if name == "set_credential":
        label, shown = "🔐 saving a secret", {"name": args.get("name"), "value": "••••••"}
    elif name in ("read_url", "load_openapi"):
        label, shown = f"🔎 {name}: {args.get('url')}", args
    else:
        label, shown = f"📡 {args.get('method')} {args.get('url')}", args
    with st.expander(label):
        st.json(shown)


# ---------------------------------------------------------------- AI layer
def ask_ai(provider, ai_key, model, base_url, system, history, use_tools=True):
    if provider["type"] == "anthropic":
        client = anthropic.Anthropic(api_key=ai_key)
        msgs = list(history)
        for _ in range(10):
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

    client = OpenAI(api_key=ai_key, base_url=base_url or None)
    msgs = [{"role": "system", "content": system}] + list(history)
    for _ in range(10):
        kwargs = dict(model=model, messages=msgs)
        if use_tools:
            kwargs["tools"] = OPENAI_TOOLS
        r = client.chat.completions.create(**kwargs)
        m = r.choices[0].message
        if not m.tool_calls:
            return m.content or ""
        msgs.append({
            "role": "assistant",
            "content": m.content or "",
            "tool_calls": [{
                "id": tc.id, "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            } for tc in m.tool_calls],
        })
        for tc in m.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            show_tool(tc.function.name, args)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": dispatch(tc.function.name, args)})
    return "(too many steps)"


def build_system():
    s = (
        "You are a friendly assistant. You talk with the user in a normal chat (reply in "
        "the language the user writes in, Bengali if they write Bengali) and you set up "
        "and run REST API calls for them using your tools. A typical request: the user "
        "pastes an API documentation link (for example a phone / telephony provider) and "
        "wants to make an outbound phone call automatically through REST. "
        "How to work: 1) read_url the documentation link. 2) Try load_openapi, and follow "
        "the relevant doc links with read_url until you find the exact endpoint for what "
        "the user wants, its required parameters, the auth style and the body format. "
        "3) Explain in very simple words what is needed (account, API key or token, from "
        "number, to number, ...) and ask only for what is still missing. 4) When the user "
        "gives secrets, store them with set_credential and never repeat them. "
        "5) Use call_api to do it. Never invent endpoints or parameters: only use what you "
        "actually read. Outbound calls cost money and contact real people, so before any "
        "POST, PUT, PATCH or DELETE show exactly what will be sent and wait for the user's "
        "yes. Keep answers short and simple."
    )
    creds = list(st.session_state.credentials.keys())
    s += f"\nStored credential names: {creds if creds else 'none'}."
    cfg = st.session_state.get("config")
    if cfg:
        s += "\nCurrently loaded API: " + json.dumps(cfg, ensure_ascii=False)[:12000]
    return s


# ---------------------------------------------------------------- UI
st.session_state.setdefault("history", [])
st.session_state.setdefault("credentials", {})

st.title("💬 API Chat Assistant")
st.caption("আমার সাথে সাধারণ চ্যাটের মতো কথা বলুন। API documentation এর লিংক চ্যাটেই দিন, বাকিটা আমি পিছনে দেখে নেব।")

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
        "যেমন লিখতে পারেন: \"এই documentation পড়ে বলো কীভাবে REST দিয়ে outbound phone call করা যায়: "
        "https://...\""
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
        st.write(final)
    st.session_state.history.append({"role": "assistant", "content": final})
