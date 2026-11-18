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
