"""Shared OpenAI HTTP envelope, with engine-specific identity/token obligations."""
import hashlib


def valid_ids(values):
    if not isinstance(values,list) or not all(type(i) is int and i >= 0 for i in values):
        raise ValueError("invalid engine token IDs")
    return values


def parse(events, body, *, strong):
    chat = "messages" in body
    streamed = body["stream"]
    ids, echoes, texts, usage, finishes = [], [], [], [], []
    response_ids, models = set(), set()
    done = False
    first = visible = terminal = None
    for ns,obj in events:
        if done:
            raise ValueError("event after terminal SSE marker")
        if obj == "[DONE]":
            if not streamed:
                raise ValueError("unexpected terminal marker")
            done = True
            terminal = ns
            continue
        if obj.get("error"):
            raise ValueError(f"engine error in response: {obj['error']}")
        response_ids.add(obj["id"])
        models.add(obj["model"])
        if obj.get("usage") is not None:
            usage.append(obj["usage"])
        if obj.get("prompt_token_ids") is not None:
            echoes.append(valid_ids(obj["prompt_token_ids"]))
        for choice in obj["choices"]:
            if choice["index"] != 0:
                raise ValueError("unexpected choice index")
            if choice.get("prompt_token_ids") is not None:
                echoes.append(valid_ids(choice["prompt_token_ids"]))
            if chat:
                message = choice["delta" if streamed else "message"]
                txt = message.get("content") or ""
                reasoning = message.get("reasoning") or message.get("reasoning_content")
            else:
                txt = choice["text"]
                reasoning = None
            chunk = valid_ids(choice.get("token_ids") or [])
            if finishes and (chunk or txt or reasoning):
                raise ValueError("generated content after finish")
            ids.extend(chunk)
            texts.append(txt)
            if first is None and (txt or reasoning or chunk):
                first = ns
            if visible is None and txt:
                visible = ns
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
        if not streamed:
            terminal = ns
    if streamed != done or len(usage) != 1 or len(finishes)!=1 or finishes[0] not in {"length","stop"}:
        raise ValueError("incomplete response, finish or usage")
    if len(response_ids)!=1 or models!={body["model"]}:
        raise ValueError("response model or request identity mismatch")
    rid = response_ids.pop()
    if not isinstance(rid,str) or not rid or (strong and not rid.endswith(body["request_id"])):
        raise ValueError("response execution ID mismatch")
    counters = usage[0]
    n, o = counters["prompt_tokens"], counters["completion_tokens"]
    if not (type(n) is int and type(o) is int and n>0 and 0<o<=body["max_tokens"]
            and counters["total_tokens"]==n+o):
        raise ValueError("invalid server usage or output budget")
    if (ids or strong and not chat) and len(ids)!=o:
        raise ValueError("generated IDs and server usage differ")
    if len(echoes)>1 or (echoes and len(echoes[0])!=n):
        raise ValueError("prompt ID echo and server usage differ")
    if strong and not chat:
        if not echoes or (isinstance(body["prompt"],list) and echoes[0]!=body["prompt"]):
            raise ValueError("prompt ID echo differs from submitted input")
    return dict(input_tokens=n,output_tokens=o,output_ids=ids or None,prompt_ids=echoes[0] if echoes else None,
                response_id=rid,request_identity_grade="engine_request_id" if strong else "local_transport_binding",
                count_source="engine_token_ids" if ids else "server_usage",finish=finishes[0],
                text_sha256=hashlib.sha256("".join(texts).encode()).hexdigest(),
                first_event_ns=first,first_visible_ns=visible,terminal_ns=terminal)
