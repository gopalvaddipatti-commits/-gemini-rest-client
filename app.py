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
            where = "query" if s.get("in") == "
