"""Deterministic request identities and planned waves. All sizes are tokenizer IDs."""
from .engines.openai import valid_ids


def plan_waves(suite):
    name=suite["suite_id"]
    waves=[]
    def cases(kind,n,o,indices,r,role="formal",stream=True,conc=1):
        wave=[]
        for i in indices:
            key=f"{kind}-r{r}-n{n}-o{o}-i{i}-c{conc}-s{int(stream)}"
            wave.append(dict(key=key,role=role,kind=kind,n=n,o=o,ix=i,round=r,concurrency=conc,stream=stream))
        waves.append(wave)
    if name in {"new-target","legacy-regression"}:
        for r in range(suite["rounds"]):
            for i in range(suite["warmup_per_round"]):
                cases("warmup",128,16,[i],r,"warmup")
            for n in suite["input_tokens"]:
                for o in suite["max_output_tokens"]:
                    for i in range(suite["texts_per_cell"]):
                        cases("fixed",n,o,[i],r)
    elif name=="same-content":
        for r in range(suite["rounds"]):
            for i in range(suite["texts_per_round"]):
                cases("same-content",None,suite["max_output_tokens"],[i],r)
    elif name=="bounded-concurrency":
        for c in suite["clients"]:
            for slot in range(0,suite["requests_per_client_setting"],c):
                cases("concurrency",suite["input_tokens"][0],suite["max_output_tokens"][0],range(slot,slot+c),0,conc=c)
    elif name in {"chat-pair","protocol-pilot","soak","rejection-recovery"}:
        if name=="soak":
            if suite["duration_s"]!=300 or suite["interval_s"]!=15:
                raise ValueError("soak must cover 300 seconds at 15-second intervals")
            for i in range(20):
                cases("soak",128,16,[0],i,"diagnostic")
        elif name=="rejection-recovery":
            cases("rejection",128,16,[0],0,"diagnostic",False)
            cases("followup",128,16,[0],0,"diagnostic")
        else:
            kind="chat" if name=="chat-pair" else "pilot"
            for stream in (True,False):
                cases(kind,None if kind=="chat" else 128,suite.get("output_budget",64 if kind=="chat" else 16),
                      [0],0,"diagnostic",stream)
    else:
        raise ValueError("unknown request suite: "+name)
    planned=[s for w in waves for s in w]
    if len(set(s["key"] for s in planned))!=len(planned):
        raise ValueError("duplicate planned task")
    if sum(s["role"]=="formal" for s in planned)!=suite.get("expected_formal_requests",0):
        raise ValueError("formal matrix differs from declared size")
    return waves


def native(target):
    adapter=target["engine"]["adapter"]
    if adapter not in {"llama_cpp.native.v1","vllm.local.v1"}:
        raise ValueError("unsupported request engine: "+adapter)
    return adapter=="llama_cpp.native.v1"


def tokenize_body(target,text):
    if native(target):
        return {"content":text,"add_special":False,"parse_special":False}
    return {"model":target["deployment"]["served_model"],"prompt":text,"add_special_tokens":False}


def build(plan,spec,execution):
    target,suite=plan["target"],plan["suite"]
    model=target["deployment"]["served_model"]
    text=suite["corpus"]["texts"][spec["ix"]]
    sampled={"temperature":0,"seed":suite.get("sampling",{}).get("seed",381127),"stream":spec["stream"]}
    if spec["kind"]=="chat":
        body=dict(model=model,messages=[{"role":"user","content":text}],max_tokens=spec["o"],**sampled)
        kwargs=target["model"].get("tokenizer",{}).get("chat_template_kwargs")
        if kwargs:
            body["chat_template_kwargs"]=kwargs
        route="/v1/chat/completions"
    else:
        if spec["n"] is None:
            prompt=text
        else:
            ids=valid_ids(plan["tokenized"][spec["ix"]])
            if not ids:
                raise ValueError("empty source tokens")
            prompt=(ids*((spec["n"]+len(ids)-1)//len(ids)))[:spec["n"]]
        if native(target):
            if spec["kind"]=="rejection":
                # Documented prompt grammar rejects an object with no prompt_string.
                prompt={"invalid_prompt_field":True}
            body=dict(model=model,prompt=prompt,n_predict=spec["o"],return_tokens=True,cache_prompt=True,**sampled)
            return "/completion",body
        body=dict(model=model,prompt=prompt,max_tokens=spec["o"],add_special_tokens=False,**sampled)
        route="/v1/completions"
    if not native(target):
        body.update(request_id=execution,return_token_ids=True)
    if spec["stream"]:
        body["stream_options"]={"include_usage":True}
    return route,body
