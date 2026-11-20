"""Pinned llama-server native completion contract. Chat uses its existing OpenAI endpoint."""
import hashlib
import math

from .openai import valid_ids


def parse(events, body, model):
    ids, texts = [], []
    visible = first = final = terminal = None
    predictions = []
    for ns, obj in events:
        if final is not None or not isinstance(obj, dict) or type(obj.get("stop")) is not bool:
            raise ValueError("native completion has invalid or duplicate terminal event")
        if obj["index"] != 0:
            raise ValueError("native choice index mismatch")
        if obj["stop"]:
            final, terminal = obj, ns
            if final.get("model") != model or final.get("truncated") is not False:
                raise ValueError("wrong native model or truncated context")
        # Partial id_slot is not the slot identity on all server versions. Native IDs
        # are bound by the request's local connection; the final slot is diagnostic.
        content = obj["content"]
        chunk = valid_ids(obj["tokens"])
        if body["stream"] and obj["stop"] and (content or chunk):
            raise ValueError("pinned native final event repeated generated data")
        if first is None and (content or chunk):
            first = ns
        if visible is None and content:
            visible = ns
        texts.append(content)
        ids.extend(chunk)
        predictions.append(obj["tokens_predicted"])
    if final is None or final.get("stop_type") not in {"limit","eos","word"}:
        raise ValueError("native terminal missing")
    n,o = final["tokens_evaluated"],final["tokens_predicted"]
    if not (type(n) is int and n>0 and type(o) is int and 0<o<=body["n_predict"] and len(ids)==o
            and predictions==sorted(predictions) and (not isinstance(body["prompt"],list) or n==len(body["prompt"]))):
        raise ValueError("native final token counts disagree with measured inputs/IDs")
    t=final["timings"]
    if not (t["predicted_n"]==o and 0<=t["prompt_n"]<=n and all(isinstance(t[k],(int,float)) and
            math.isfinite(t[k]) and t[k]>=0 for k in ("prompt_ms","predicted_ms"))):
        raise ValueError("native request timing or count mismatch")
    phases={key:t[key] for key in ("prompt_ms","predicted_ms","prompt_n","predicted_n")}
    phases.update(attribution="request-local native timings",cache_n=t.get("cache_n"),
                  tokens_cached_raw=final.get("tokens_cached"),id_slot=final.get("id_slot"))
    return dict(input_tokens=n,output_tokens=o,output_ids=ids,prompt_ids=None,response_id=None,
                request_identity_grade="local_transport_binding",count_source="engine_token_ids",finish=final["stop_type"],
                text_sha256=hashlib.sha256("".join(texts).encode()).hexdigest(),
                first_event_ns=first,first_visible_ns=visible,terminal_ns=terminal,engine=phases)
