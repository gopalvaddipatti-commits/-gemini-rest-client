import json
import re
from urllib.parse import urljoin, urlparse

import anthropic
import requests
import streamlit as st
import yaml
from bs4 import BeautifulSoup
from openai import OpenAI

st.set_page_config(page_title="API Chat (any AI)", page_icon="🔌")

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
    "Custom (OpenAI-compatible)": {"type": "openai", "base_url": ""},
}

TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "method": {"type": "string", "description": "GET, POST, PUT, PATCH or DELETE"},
        "path": {"type": "string", "description": "Endpoint path, e.g. /users/{id}"},
        "params": {"type": "object", "description": "Query or path parameters"},
        "body": {"type": "object", "description": "JSON body for POST/PUT/PATCH"},
    },
    "required": ["method", "path"],
}
TOOL_DESC = "Send a real HTTP request to the connected target API. Pick the endpoint from the list in the system prompt."

ANTHROPIC_TOOLS = [{"name": "call_api", "description": TOOL_DESC, "input_schema": TOOL_SCHEMA}]
OPENAI_TOOLS = [{
    "type": "function",
    "function": {"name": "call_api", "description": TOOL_DESC, "parameters": TOOL_SCHEMA},
}]


# ---------------------------------------------------------------- docs -> config
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
    # 1) the url itself may be the spec
    spec = try_load_spec(doc_url)
    if spec:
        return spec
    # 2) links on the page
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
    # 3) common paths
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

    auth = {"type": "bearer", "header": "Authorization"}
    schemes = (spec.get("components", {}).get("securitySchemes")
               or spec.get("securityDefinitions") or {})
    for s in schemes.values():
        if s.get("type") == "apiKey":
            where = "query" if s.get("in") == "query" else "header"
            auth = {"type": where, "header": s.get("name", "X-API-Key")}
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
                "method": method.upper(),
                "path": path,
                "summary": info.get("summary", ""),
                "params": params,
            })
    return {"base_url": base_url, "auth": auth, "endpoints": endpoints}


# ---------------------------------------------------------------- target API call
def send_request(config, api_key, method, path, params, body):
    url = config["base_url"].rstrip("/") + path
    headers = {}
    query = dict(params or {})
    auth = config.get("auth", {})
    if api_key:
        if auth.get("type") == "header":
            headers[auth.get("header", "X-API-Key")] = api_key
        elif auth.get("type") == "query":
            query[auth.get("header", "api_key")] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    return requests.request(
        method.upper(), url, headers=headers, params=query,
        json=body if body else None, timeout=30,
    )


def run_tool(config, api_key, tool_input):
    path = tool_input["path"]
    params = dict(tool_input.get("params") or {})
    for k in list(params.keys()):
        if "{" + k + "}" in path:
            path = path.replace("{" + k + "}", str(params.pop(k)))
    try:
        resp = send_request(config, api_key, tool_input["method"], path,
                            params, tool_input.get("body"))
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        out = {"status": resp.status_code, "data": data}
    except Exception as e:
        out = {"error": str(e)}
    return json.dumps(out, ensure_ascii=False)[:20000]


# ---------------------------------------------------------------- AI layer
def ask_ai(provider, ai_key, model, base_url, system, history, config=None, api_key=""):
    """history = list of {role, content(str)}. Returns final text."""
    config = config or {}
    if provider["type"] == "anthropic":
        client = anthropic.Anthropic(api_key=ai_key)
        msgs = list(history)
        for _ in range(6):
            kwargs = dict(model=model, max_tokens=2000, system=system, messages=msgs)
            if config:
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
                    with st.expander(f"API call: {b.input.get('method')} {b.input.get('path')}"):
                        st.json(b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": run_tool(config, api_key, b.input)})
            msgs.append({"role": "user", "content": results})
        return "(too many tool calls)"

    # OpenAI-compatible: OpenAI, Grok, DeepSeek, Gemini, custom
    client = OpenAI(api_key=ai_key, base_url=base_url or None)
    msgs = [{"role": "system", "content": system}] + list(history)
    for _ in range(6):
        kwargs = dict(model=model, messages=msgs)
        if config:
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
            with st.expander(f"API call: {args.get('method')} {args.get('path')}"):
                st.json(args)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": run_tool(config, api_key, args)})
    return "(too many tool calls)"


def extract_with_ai(provider, ai_key, model, base_url, doc_url):
    html = requests.get(doc_url, headers=HEADERS, timeout=20).text
    text = BeautifulSoup(html, "html.parser").get_text("\n")[:60000]
    prompt = (
        "Below is API documentation text. Return ONLY valid JSON, no explanation, in this shape: "
        '{"base_url": str, "auth": {"type": "bearer|header|query", "header": str}, '
        '"endpoints": [{"method": str, "path": str, "summary": str, "params": [str]}]}\n\n'
        + text
    )
    raw = ask_ai(provider, ai_key, model, base_url, "Return JSON only.",
                 [{"role": "user", "content": prompt}])
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


# ---------------------------------------------------------------- UI
st.title("🔌 API Chat (any AI)")

with st.sidebar:
    st.subheader("1. AI provider")
    pname = st.selectbox("Provider", list(PROVIDERS.keys()))
    provider = PROVIDERS[pname]
    base_url = provider["base_url"]
    if pname.startswith("Custom"):
        base_url = st.text_input("Base URL", placeholder="https://.../v1")
    model = st.text_input("Model name", help="Provider এর docs/console থেকে দেখে বসান")
    ai_key = st.text_input("AI API key", type="password")

    if st.button("Test this AI key"):
        if not ai_key or not model:
            st.warning("Model name আর AI key দিন")
        else:
            try:
                reply = ask_ai(provider, ai_key, model, base_url, "Reply with one word.",
                               [{"role": "user", "content": "Say hi"}])
                st.success(f"Key works: {reply[:80]}")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.subheader("2. Target API")
    doc_url = st.text_input("Documentation URL")
    target_key = st.text_input("Target API key", type="password")

    if st.button("Connect") and doc_url:
        try:
            with st.spinner("OpenAPI খুঁজছি..."):
                spec = find_openapi(doc_url)
            if spec:
                st.session_state.config = spec_to_config(spec, doc_url)
                st.success("OpenAPI spec পাওয়া গেছে")
            else:
                if not ai_key or not model:
                    st.error("Spec পাওয়া যায়নি। Docs পড়ার জন্য AI key আর model name দিন।")
                    st.stop()
                with st.spinner("AI docs পড়ছে..."):
                    st.session_state.config = extract_with_ai(
                        provider, ai_key, model, base_url, doc_url)
                st.success("AI দিয়ে docs পড়া হয়েছে")
            st.session_state.history = []
        except Exception as e:
            st.error(f"Connect failed: {e}")

config = st.session_state.get("config")
if not config:
    st.info("Sidebar এ AI provider আর documentation link দিয়ে Connect চাপুন।")
    st.stop()

with st.expander(f"Connected: {config.get('base_url', '')} ({len(config.get('endpoints', []))} endpoints)"):
    st.json(config)

st.session_state.setdefault("history", [])

SYSTEM = (
    "You are an API assistant. Call the target API for the user with the call_api tool. "
    "Base URL: " + config.get("base_url", "") + "\nEndpoints:\n"
    + json.dumps(config.get("endpoints", []), ensure_ascii=False)[:30000]
    + "\nReply in the user's language and explain results simply. "
    "Ask the user to confirm before any POST, PUT, PATCH or DELETE."
)

for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

user_text = st.chat_input("কী API কল করতে চান, লিখুন...")
if user_text:
    st.session_state.history.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.write(user_text)
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                final = ask_ai(provider, ai_key, model, base_url, SYSTEM,
                               st.session_state.history, config, target_key)
            except Exception as e:
                final = f"Error: {e}"
        st.write(final)
    st.session_state.history.append({"role": "assistant", "content": final})
