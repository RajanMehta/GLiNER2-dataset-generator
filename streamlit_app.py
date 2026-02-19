import json
import re
from typing import Any

import streamlit as st
from openai import OpenAI
from ollama import Client as OllamaClient

# ==== PAGE CONFIG ====
st.set_page_config(
    page_title="GLiNER2 Dataset Generator",
    page_icon="🏷️",
    layout="wide",
)

# ==== SESSION STATE DEFAULTS ====
# Must run before any form renders so that defaults are set on first load
# but never overwrite user edits on subsequent reruns.

st.session_state.setdefault("ner_entity_types", [
    {"name": "account_type", "description": "Type of bank account mentioned (e.g. checking, savings, credit card)"},
    {"name": "amount",       "description": "Dollar amount mentioned in the request"},
])

st.session_state.setdefault("clf_tasks", [
    {
        "task_name":         "intent",
        "labels":            "check_balance, make_transfer, transaction_history, dispute_charge",
        "multi_label":       False,
        "label_descriptions": "",
        "custom_prompt":     "",
        "few_shot_examples": "",
    }
])

st.session_state.setdefault("json_structures", [
    {
        "parent_name":    "transfer",
        "allow_multiple": False,
        "fields": [
            {"name": "from_account", "field_type": "choice", "choices": "checking, savings, credit card", "description": "Account to transfer funds from"},
            {"name": "to_account",   "field_type": "choice", "choices": "checking, savings, credit card", "description": "Account to transfer funds to"},
            {"name": "amount",       "field_type": "string", "choices": "", "description": "Dollar amount to transfer"},
        ],
    }
])

st.session_state.setdefault("gen_cache", None)

# ==== LLM CALLER ====

def make_llm_caller(mode: str, api_key: str, base_url: str, model: str,
                    temperature: float, top_p: float):
    """Returns a callable call_llm(system, user) -> str for OpenAI or Ollama."""
    if mode == "OpenAI":
        client = OpenAI(api_key=api_key)

        def call_llm(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                top_p=top_p,
            )
            return resp.choices[0].message.content

    else:
        # Normalise host: strip any path suffix the user may have copy-pasted
        _host = base_url.rstrip("/")
        for _suffix in ("/v1", "/api/generate", "/api"):
            if _host.endswith(_suffix):
                _host = _host[: -len(_suffix)]
                break

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        ollama_client = OllamaClient(host=_host, headers=headers)

        def call_llm(system: str, user: str) -> str:
            resp = ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                options={"temperature": temperature, "top_p": top_p},
            )
            return resp.message.content

    return call_llm


# ==== PROMPT BUILDERS ====

SYSTEM_PROMPT = (
    "You are a training-data annotator. "
    "Return ONLY a valid JSON array — no markdown, no code fences, no commentary. "
    "Each element must follow the exact schema described in the user message. "
    "Start your response with [ and end with ]."
)


def _seed_block(seeds: list[str]) -> str:
    if not seeds:
        return ""
    examples = "\n".join(f'  - "{s}"' for s in seeds[:8])
    return f"\nStyle reference — generate texts in a similar style to these:\n{examples}\n"


def _ner_prompt_section(ner_config: dict) -> str:
    types = ner_config["entity_types"]
    type_lines = "\n".join(
        f'    - "{et["name"]}"{(" — " + et["description"]) if et["description"] else ""}'
        for et in types
    )
    type_names = [et["name"] for et in types]
    example_entities = json.dumps(
        [{"type": t, "mention": f"<{t} mention>"} for t in type_names[:3]], ensure_ascii=False
    )
    return f"""
=== NER (Named Entity Recognition) ===
Each item must have:
  "text": a realistic text passage (1–3 sentences)
  "entities": a list of objects, each with:
    "type": one of {json.dumps(type_names)}
    "mention": the exact substring of "text" where that entity appears

Allowed entity types:
{type_lines}

Rules:
- EVERY "mention" must appear verbatim (character-for-character) inside "text".
- If an entity type has no mentions in a given text, omit it from the list (do not include it with an empty mention).
- A single text may contain zero or many entities.

Example element:
{{"text": "...", "entities": {example_entities}}}
"""


def _clf_prompt_section(clf_config: dict) -> str:
    sections = []
    for task in clf_config["tasks"]:
        name    = task["task_name"]
        labels  = task["labels"]
        multi   = task["multi_label"]
        prompt_ = task.get("custom_prompt", "")
        descs   = task.get("label_descriptions", {})
        shots   = task.get("few_shot_examples", [])

        label_str = json.dumps(labels)
        if multi:
            key_desc = (
                f'  "labels": a JSON list of one or more values from {label_str} '
                f'that apply to the text (multi-label allowed)'
            )
            example_val = f'"labels": ["{labels[0]}", "{labels[1] if len(labels) > 1 else labels[0]}"]'
        else:
            key_desc = f'  "label": exactly one value from {label_str}'
            example_val = f'"label": "{labels[0]}"'

        desc_block = ""
        if descs:
            desc_lines = "\n".join(f'    - "{k}": {v}' for k, v in descs.items())
            desc_block = f"\n  Label meanings:\n{desc_lines}"

        prompt_block = f'\n  Task instruction: "{prompt_}"' if prompt_ else ""

        shot_block = ""
        if shots:
            shot_lines = "\n".join(
                f'    input: "{inp}" → label: "{lbl}"' for inp, lbl in shots[:3]
            )
            shot_block = f"\n  Few-shot examples:\n{shot_lines}"

        sections.append(
            f'Task "{name}":\n{key_desc}{desc_block}{prompt_block}{shot_block}\n'
            f'  Example element: {{"text": "...", {example_val}}}'
        )

    joined = "\n\n".join(sections)
    return f"\n=== Classification ===\n{joined}\n"


def _json_prompt_section(json_config: dict) -> str:
    sections = []
    for struct in json_config["structures"]:
        parent    = struct["parent_name"]
        multi     = struct["allow_multiple"]
        fields    = struct["fields"]

        field_lines = []
        for f in fields:
            desc = f" — {f['description']}" if f["description"] else ""
            if f["field_type"] == "choice":
                choices = json.dumps([c.strip() for c in f["choices"].split(",") if c.strip()])
                field_lines.append(f'    "{f["name"]}" (choose one from {choices}){desc}')
            elif f["field_type"] == "list":
                field_lines.append(f'    "{f["name"]}" (list of strings){desc}')
            else:
                field_lines.append(f'    "{f["name"]}" (string){desc}')

        fields_block = "\n".join(field_lines)

        if multi:
            example_val = (
                f'"instances": [{{'
                + ", ".join(f'"{f["name"]}": "..."' for f in fields[:3])
                + "}}, ...]"
            )
            instance_note = (
                '  "instances": a JSON list of objects (one per occurrence in the text). '
                "Include multiple when the text mentions several."
            )
        else:
            example_val = (
                '"fields": {'
                + ", ".join(f'"{f["name"]}": "..."' for f in fields[:3])
                + "}"
            )
            instance_note = '  "fields": a single JSON object with the fields below.'

        sections.append(
            f'Structure "{parent}":\n{instance_note}\n  Field names and types:\n{fields_block}\n'
            f'  Example element: {{"text": "...", {example_val}}}'
        )

    joined = "\n\n".join(sections)
    return f"\n=== JSON Extraction ===\n{joined}\n"


def _multi_clf_prompt_section(clf_config: dict) -> str:
    """Variant of CLF section for multi-task mode: LLM returns a 'classifications' dict."""
    task_lines = []
    for task in clf_config["tasks"]:
        name   = task["task_name"]
        labels = task["labels"]
        multi  = task["multi_label"]
        if multi:
            task_lines.append(f'  "{name}": list of values from {json.dumps(labels)}')
        else:
            task_lines.append(f'  "{name}": one value from {json.dumps(labels)}')

    joined = "\n".join(task_lines)
    example_clf = json.dumps({t["task_name"]: t["labels"][0] for t in clf_config["tasks"]})
    return f"""
=== Classification ===
Each item must include a "classifications" key — a JSON object where:
{joined}

Example: "classifications": {example_clf}
"""


def build_user_prompt(config: dict) -> str:
    n        = config["n_examples"]
    domain   = config["domain"] or "general text"
    seeds    = config["seed_texts"]
    tasks    = config["tasks"]
    multi_task = len(tasks) > 1

    parts = [
        f"Generate exactly {n} diverse training examples for the domain: \"{domain}\".",
        _seed_block(seeds),
    ]

    # Task-specific instructions
    if "ner" in tasks:
        parts.append(_ner_prompt_section(tasks["ner"]))

    if "clf" in tasks:
        if multi_task:
            parts.append(_multi_clf_prompt_section(tasks["clf"]))
        else:
            parts.append(_clf_prompt_section(tasks["clf"]))

    if "json" in tasks:
        parts.append(_json_prompt_section(tasks["json"]))

    # Footer
    if multi_task:
        # List the actual JSON field names described in the sections above, not the internal task keys
        key_map = {"ner": '"entities"', "clf": '"classifications"', "json": '"fields" or "instances"'}
        active = ", ".join(key_map[k] for k in tasks if k in key_map)
        parts.append(
            f"\nEach element must contain: \"text\" and all task-specific fields described above ({active})."
            f"\nReturn a JSON array of exactly {n} objects. Begin with ["
        )
    else:
        parts.append(f"\nReturn a JSON array of exactly {n} objects. Begin with [")

    return "\n".join(parts)


# ==== DETERMINISTIC CONVERTERS ====

def convert_ner(item: dict, ner_config: dict) -> dict:
    entities: dict[str, list[str]] = {}
    for e in item.get("entities", []):
        etype   = (e.get("type") or "").strip()
        mention = (e.get("mention") or "").strip()
        if etype and mention:
            entities.setdefault(etype, []).append(mention)

    output: dict[str, Any] = {"entities": entities}

    descs = {
        et["name"]: et["description"]
        for et in ner_config["entity_types"]
        if et["description"]
    }
    if descs:
        output["entity_descriptions"] = descs

    return {"input": item["text"], "output": output}


def convert_clf(item: dict, clf_task_config: dict) -> dict:
    raw = item.get("labels") or item.get("label") or []
    true_label = raw if isinstance(raw, list) else [raw]
    true_label = [str(l) for l in true_label if l]

    clf_entry: dict[str, Any] = {
        "task":       clf_task_config["task_name"],
        "labels":     clf_task_config["labels"],
        "true_label": true_label,
    }
    if clf_task_config["multi_label"]:
        clf_entry["multi_label"] = True
    if clf_task_config.get("label_descriptions"):
        clf_entry["label_descriptions"] = clf_task_config["label_descriptions"]
    if clf_task_config.get("custom_prompt"):
        clf_entry["prompt"] = clf_task_config["custom_prompt"]
    if clf_task_config.get("few_shot_examples"):
        clf_entry["examples"] = clf_task_config["few_shot_examples"]

    return {"input": item["text"], "output": {"classifications": [clf_entry]}}


def convert_json_extraction(item: dict, struct_config: dict) -> dict:
    parent  = struct_config["parent_name"]
    fields  = struct_config["fields"]
    multi   = struct_config["allow_multiple"]

    def format_fields(raw: dict) -> dict:
        result = {}
        for f in fields:
            val = raw.get(f["name"], "")
            if f["field_type"] == "choice":
                choices = [c.strip() for c in f["choices"].split(",") if c.strip()]
                result[f["name"]] = {"value": str(val) if val else "", "choices": choices}
            elif f["field_type"] == "list":
                result[f["name"]] = val if isinstance(val, list) else ([val] if val else [])
            else:
                result[f["name"]] = str(val) if val else ""
        return result

    if multi and "instances" in item:
        structures = [{parent: format_fields(inst)} for inst in item["instances"]]
    else:
        structures = [{parent: format_fields(item.get("fields", {}))}]

    output: dict[str, Any] = {"json_structures": structures}

    field_descs = {f["name"]: f["description"] for f in fields if f["description"]}
    if field_descs:
        output["json_descriptions"] = {parent: field_descs}

    return {"input": item["text"], "output": output}


def convert_multi_task(item: dict, config: dict) -> dict:
    tasks = config["tasks"]
    combined: dict[str, Any] = {}

    if "ner" in tasks:
        # Accept "entities" (correct) or "ner" (LLM shorthand) as the list key
        entities_raw = item.get("entities") or item.get("ner") or []
        ner_item = {**item, "entities": entities_raw}
        ner_out = convert_ner(ner_item, tasks["ner"])["output"]
        combined.update(ner_out)

    if "clf" in tasks:
        # Accept "classifications" (correct) or "clf" (LLM shorthand) as the dict key
        clf_dict = item.get("classifications") or item.get("clf") or {}
        if not isinstance(clf_dict, dict):
            clf_dict = {}
        all_clf_entries = []
        for task_cfg in tasks["clf"]["tasks"]:
            name = task_cfg["task_name"]
            raw_label = clf_dict.get(name)
            if raw_label is None:
                continue
            true_label = raw_label if isinstance(raw_label, list) else [raw_label]
            true_label = [str(l) for l in true_label if l]
            entry: dict[str, Any] = {
                "task":       name,
                "labels":     task_cfg["labels"],
                "true_label": true_label,
            }
            if task_cfg["multi_label"]:
                entry["multi_label"] = True
            if task_cfg.get("label_descriptions"):
                entry["label_descriptions"] = task_cfg["label_descriptions"]
            if task_cfg.get("custom_prompt"):
                entry["prompt"] = task_cfg["custom_prompt"]
            if task_cfg.get("few_shot_examples"):
                entry["examples"] = task_cfg["few_shot_examples"]
            all_clf_entries.append(entry)
        if all_clf_entries:
            combined["classifications"] = all_clf_entries

    if "json" in tasks:
        struct = tasks["json"]["structures"][0]
        json_out = convert_json_extraction(item, struct)["output"]
        combined.update(json_out)

    return {"input": item["text"], "output": combined}


# ==== VALIDATION ====

def validate_and_convert(raw_list: list, config: dict) -> tuple[list, list]:
    """
    Returns (valid_gliner2_examples, warning_messages).
    Structure is guaranteed by converters; we only check values here.
    """
    tasks   = config["tasks"]
    multi   = len(tasks) > 1
    valid   = []
    warnings = []

    for idx, item in enumerate(raw_list):
        label = f"Item {idx + 1}"

        if not isinstance(item, dict):
            warnings.append(f"{label}: not a dict — skipped")
            continue
        if not item.get("text") or not isinstance(item["text"], str):
            warnings.append(f"{label}: missing or empty 'text' — skipped")
            continue

        text = item["text"]

        # Convert to GLiNER2 format
        try:
            if multi:
                gliner_ex = convert_multi_task(item, config)
            elif "ner" in tasks:
                gliner_ex = convert_ner(item, tasks["ner"])
            elif "clf" in tasks:
                task_cfg = tasks["clf"]["tasks"][0]
                gliner_ex = convert_clf(item, task_cfg)
            elif "json" in tasks:
                struct = tasks["json"]["structures"][0]
                gliner_ex = convert_json_extraction(item, struct)
            else:
                warnings.append(f"{label}: no active task — skipped")
                continue
        except Exception as e:
            warnings.append(f"{label}: conversion error — {e} — skipped")
            continue

        out = gliner_ex.get("output", {})

        # NER: soft-warn if any mention not found verbatim in text
        if "entities" in out:
            text_lower = text.lower()
            for etype, mentions in out["entities"].items():
                for mention in mentions:
                    if mention and mention.lower() not in text_lower:
                        warnings.append(
                            f"{label}: NER mention '{mention}' ({etype}) not found in text"
                        )

        # CLF: soft-warn if true_label not in labels
        if "classifications" in out:
            for entry in out["classifications"]:
                allowed = set(entry.get("labels", []))
                for lbl in entry.get("true_label", []):
                    if lbl and lbl not in allowed:
                        warnings.append(
                            f"{label}: CLF task '{entry.get('task')}' — "
                            f"true_label '{lbl}' not in labels"
                        )

        valid.append(gliner_ex)

    return valid, warnings


# ==== EXPORT ====

def to_jsonl(examples: list) -> bytes:
    lines = [json.dumps(ex, ensure_ascii=False) for ex in examples]
    return "\n".join(lines).encode("utf-8")


# ==== CONFIG ASSEMBLY ====

def collect_config() -> dict:
    tasks: dict[str, Any] = {}

    if st.session_state.get("use_ner"):
        entity_types = [
            {"name": r["name"].strip(), "description": r["description"].strip()}
            for r in st.session_state["ner_entity_types"]
            if r["name"].strip()
        ]
        if entity_types:
            tasks["ner"] = {"entity_types": entity_types}

    if st.session_state.get("use_clf"):
        clf_tasks = []
        for t in st.session_state["clf_tasks"]:
            if not t["task_name"].strip():
                continue
            labels = [l.strip() for l in t["labels"].split(",") if l.strip()]
            if not labels:
                continue

            # Parse "label: description" lines
            descs: dict[str, str] = {}
            for line in t["label_descriptions"].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip():
                        descs[k.strip()] = v.strip()

            # Parse "input | label" few-shot lines
            shots: list[list[str]] = []
            for line in t["few_shot_examples"].splitlines():
                if "|" in line:
                    parts = line.split("|", 1)
                    shots.append([parts[0].strip(), parts[1].strip()])

            clf_tasks.append({
                "task_name":         t["task_name"].strip(),
                "labels":            labels,
                "multi_label":       t["multi_label"],
                "label_descriptions": descs,
                "custom_prompt":     t["custom_prompt"].strip(),
                "few_shot_examples": shots,
            })

        if clf_tasks:
            tasks["clf"] = {"tasks": clf_tasks}

    if st.session_state.get("use_json"):
        structures = []
        for s in st.session_state["json_structures"]:
            if not s["parent_name"].strip():
                continue
            fields = [
                {
                    "name":       f["name"].strip(),
                    "field_type": f["field_type"],
                    "choices":    f["choices"],
                    "description": f["description"].strip(),
                }
                for f in s["fields"]
                if f["name"].strip()
            ]
            if fields:
                structures.append({
                    "parent_name":    s["parent_name"].strip(),
                    "allow_multiple": s["allow_multiple"],
                    "fields":         fields,
                })
        if structures:
            tasks["json"] = {"structures": structures}

    seeds = [
        s.strip()
        for s in st.session_state.get("seed_texts", "").splitlines()
        if s.strip()
    ]

    return {
        "domain":     st.session_state.get("domain", "").strip(),
        "seed_texts": seeds,
        "n_examples": int(st.session_state.get("n_examples", 20)),
        "tasks":      tasks,
    }


# ==== SIDEBAR ====

with st.sidebar:
    st.header("API Settings")

    mode = st.selectbox("Provider", ["Ollama", "OpenAI"], key="llm_mode")

    if mode == "OpenAI":
        st.text_input("OpenAI API Key", type="password", key="openai_key")
        st.text_input("Model", value="gpt-4o-mini", key="openai_model")
        st.session_state.setdefault("ollama_base", "http://localhost:11434")
        st.session_state.setdefault("ollama_key", "")
        st.session_state.setdefault("ollama_model", "gpt-oss:120b")
    else:
        ollama_host = st.radio(
            "Host", ["Hosted (ollama.com)", "Local"], horizontal=True, key="ollama_host"
        )
        # When the radio changes, sync base URL and model to sensible defaults
        if st.session_state.get("_ollama_host_prev") != ollama_host:
            st.session_state["ollama_base"] = (
                "https://ollama.com" if ollama_host == "Hosted (ollama.com)" else "http://localhost:11434"
            )
            st.session_state["ollama_model"] = "gpt-oss:120b"
            st.session_state["_ollama_host_prev"] = ollama_host
        st.text_input(
            "Base URL",
            key="ollama_base",
            help="Host only — do not include /api or /v1.",
        )
        st.text_input(
            "API Key" + (" (required)" if ollama_host == "Hosted (ollama.com)" else " (optional)"),
            type="password",
            key="ollama_key",
            help="Your ollama.com Bearer token. Not needed for local Ollama.",
        )
        st.text_input("Model", key="ollama_model")
        st.session_state.setdefault("openai_key", "")
        st.session_state.setdefault("openai_model", "gpt-4o-mini")

    st.divider()
    st.slider("Temperature", 0.0, 1.5, 0.8, 0.05, key="temperature")
    st.slider("Top-p", 0.1, 1.0, 0.95, 0.05, key="top_p")
    st.number_input("Examples to generate", min_value=5, max_value=500, value=20, step=5,
                    key="n_examples")
    st.caption("Batches of 25 are used automatically for large counts.")


# ==== MAIN UI ====

st.title("GLiNER2 Training Dataset Generator")
st.caption(
    "Define your annotation schema, describe your domain, and generate annotated JSONL "
    "ready for GLiNER2 finetuning."
)

# ---- Step 1: Task selection ----
st.subheader("Step 1: Select Task Type(s)")
col1, col2, col3 = st.columns(3)
with col1:
    st.checkbox("Named Entity Recognition (NER)", value=True, key="use_ner")
with col2:
    st.checkbox("Classification (CLF)", value=False, key="use_clf")
with col3:
    st.checkbox("JSON Structure Extraction", value=False, key="use_json")

# ---- Step 2: Domain context ----
st.subheader("Step 2: Domain Context")
st.text_area(
    "Describe your domain and the kind of text to generate",
    placeholder=(
        "e.g. A financial chatbot where users ask about account balances, "
        "request transfers between accounts, or inquire about recent transactions."
    ),
    height=90,
    key="domain",
)
st.text_area(
    "Optional seed texts for style reference (one per line)",
    height=70,
    placeholder="e.g. What's my current savings account balance?\nI'd like to move $200 from checking to savings.",
    key="seed_texts",
)

# ---- Step 3: Schema forms ----
st.subheader("Step 3: Define Schema")

# -- NER form --
if st.session_state.get("use_ner"):
    with st.expander("NER Entity Types", expanded=True):
        st.caption("Define the entity types the model should extract.")
        rows: list = st.session_state["ner_entity_types"]

        for i, row in enumerate(rows):
            c1, c2, c3 = st.columns([2, 4, 0.6])
            with c1:
                row["name"] = st.text_input(
                    "Entity type", value=row["name"],
                    key=f"ner_name_{i}", placeholder="e.g. medication"
                )
            with c2:
                row["description"] = st.text_input(
                    "Description (optional)", value=row["description"],
                    key=f"ner_desc_{i}", placeholder="e.g. Names of drugs or pharmaceutical products"
                )
            with c3:
                st.write("")
                st.write("")
                if st.button("✕", key=f"ner_rm_{i}", help="Remove") and len(rows) > 1:
                    rows.pop(i)
                    st.rerun()

        if st.button("+ Add Entity Type", key="ner_add"):
            rows.append({"name": "", "description": ""})
            st.rerun()

# -- CLF form --
if st.session_state.get("use_clf"):
    with st.expander("Classification Tasks", expanded=True):
        tasks_list: list = st.session_state["clf_tasks"]

        for i, task in enumerate(tasks_list):
            st.markdown(f"**Task {i + 1}**")

            col_main, col_rm = st.columns([11, 1])
            with col_rm:
                if st.button("✕", key=f"clf_rm_{i}", help="Remove task") and len(tasks_list) > 1:
                    tasks_list.pop(i)
                    st.rerun()

            ca, cb, cc = st.columns([2, 4, 1.5])
            with ca:
                task["task_name"] = st.text_input(
                    "Task name", value=task["task_name"],
                    key=f"clf_name_{i}", placeholder="e.g. sentiment"
                )
            with cb:
                task["labels"] = st.text_input(
                    "Labels (comma-separated)", value=task["labels"],
                    key=f"clf_labels_{i}", placeholder="positive, negative, neutral"
                )
            with cc:
                task["multi_label"] = st.checkbox(
                    "Multi-label", value=task["multi_label"], key=f"clf_multi_{i}"
                )

            with st.expander("Optional: descriptions, prompt, few-shot examples"):
                task["label_descriptions"] = st.text_area(
                    "Label descriptions (one per line: label: description)",
                    value=task["label_descriptions"], height=70,
                    key=f"clf_descs_{i}",
                    placeholder="positive: Text expresses positive sentiment\nnegative: Text expresses negative sentiment",
                )
                task["custom_prompt"] = st.text_input(
                    "Custom task prompt",
                    value=task["custom_prompt"], key=f"clf_prompt_{i}",
                    placeholder="Classify the sentiment of the following customer review.",
                )
                task["few_shot_examples"] = st.text_area(
                    "Few-shot examples (input text | label, one per line)",
                    value=task["few_shot_examples"], height=70,
                    key=f"clf_shots_{i}",
                    placeholder="Great product, highly recommend! | positive\nTerrible experience. | negative",
                )

            if i < len(tasks_list) - 1:
                st.divider()

        if st.button("+ Add Classification Task", key="clf_add"):
            tasks_list.append({
                "task_name": "", "labels": "", "multi_label": False,
                "label_descriptions": "", "custom_prompt": "", "few_shot_examples": "",
            })
            st.rerun()

# -- JSON Extraction form --
if st.session_state.get("use_json"):
    with st.expander("JSON Extraction Schema", expanded=True):
        FIELD_TYPES = ["string", "list", "choice"]
        structs: list = st.session_state["json_structures"]

        for si, struct in enumerate(structs):
            st.markdown(f"**Structure {si + 1}**")

            sc1, sc2, sc3 = st.columns([3, 2.5, 0.6])
            with sc1:
                struct["parent_name"] = st.text_input(
                    "Structure name (parent key)", value=struct["parent_name"],
                    key=f"js_name_{si}", placeholder="e.g. product"
                )
            with sc2:
                struct["allow_multiple"] = st.checkbox(
                    "Allow multiple instances per text",
                    value=struct["allow_multiple"], key=f"js_multi_{si}"
                )
            with sc3:
                st.write("")
                st.write("")
                if st.button("✕", key=f"js_rm_{si}", help="Remove structure") and len(structs) > 1:
                    structs.pop(si)
                    st.rerun()

            st.caption("Fields:")
            fields_list: list = struct["fields"]

            for fi, field in enumerate(fields_list):
                fc1, fc2, fc3, fc4, fc_rm = st.columns([2, 1.3, 2, 2.5, 0.5])
                with fc1:
                    field["name"] = st.text_input(
                        "Field name", value=field["name"],
                        key=f"js_fname_{si}_{fi}", placeholder="e.g. price"
                    )
                with fc2:
                    cur_type = field.get("field_type", "string")
                    cur_idx  = FIELD_TYPES.index(cur_type) if cur_type in FIELD_TYPES else 0
                    field["field_type"] = st.selectbox(
                        "Type", FIELD_TYPES, index=cur_idx, key=f"js_ftype_{si}_{fi}"
                    )
                with fc3:
                    if field["field_type"] == "choice":
                        field["choices"] = st.text_input(
                            "Choices (comma-sep)", value=field.get("choices", ""),
                            key=f"js_fchoices_{si}_{fi}", placeholder="small, medium, large"
                        )
                    else:
                        field["choices"] = ""
                        st.empty()
                with fc4:
                    field["description"] = st.text_input(
                        "Description (optional)", value=field.get("description", ""),
                        key=f"js_fdesc_{si}_{fi}"
                    )
                with fc_rm:
                    st.write("")
                    st.write("")
                    if st.button("✕", key=f"js_frm_{si}_{fi}", help="Remove field") and len(fields_list) > 1:
                        fields_list.pop(fi)
                        st.rerun()

            if st.button(f"+ Add Field", key=f"js_fadd_{si}"):
                fields_list.append({"name": "", "field_type": "string", "choices": "", "description": ""})
                st.rerun()

            if si < len(structs) - 1:
                st.divider()

        if st.button("+ Add Structure", key="js_add"):
            structs.append({
                "parent_name": "", "allow_multiple": False,
                "fields": [{"name": "", "field_type": "string", "choices": "", "description": ""}],
            })
            st.rerun()


# ---- Step 4: Generate ----
st.subheader("Step 4: Generate")

config = collect_config()
has_task   = bool(config["tasks"])
has_domain = bool(config["domain"])

if not has_task:
    st.info("Select at least one task type and define its schema.")
elif not has_domain:
    st.info("Provide a domain description so the LLM knows what kind of text to generate.")

generate_btn = st.button(
    "Generate Dataset",
    disabled=not (has_task and has_domain),
    type="primary",
    key="generate_btn",
)

if generate_btn:
    mode_val  = st.session_state.get("llm_mode", "OpenAI")
    api_key   = st.session_state.get("openai_key", "") if mode_val == "OpenAI" else st.session_state.get("ollama_key", "")
    base_url  = st.session_state.get("ollama_base", "http://localhost:11434")
    model_val = st.session_state.get("openai_model", "gpt-4o-mini") if mode_val == "OpenAI" else st.session_state.get("ollama_model", "gpt-oss:120b")
    temp      = float(st.session_state.get("temperature", 0.8))
    topp      = float(st.session_state.get("top_p", 0.95))
    n_total   = config["n_examples"]

    try:
        call_llm = make_llm_caller(mode_val, api_key, base_url, model_val, temp, topp)
    except Exception as e:
        st.error(f"Could not create LLM client: {e}")
        st.stop()

    system_prompt = SYSTEM_PROMPT

    # Batch into chunks of 25
    batch_size = 25
    n_batches  = max(1, -(-n_total // batch_size))  # ceiling division
    all_raw: list = []

    progress = st.progress(0.0)
    status   = st.empty()

    for b in range(n_batches):
        this_n = min(batch_size, n_total - b * batch_size)
        batch_config = {**config, "n_examples": this_n}
        user_prompt  = build_user_prompt(batch_config)

        status.write(f"Generating batch {b + 1} / {n_batches} ({this_n} examples)…")
        try:
            raw_response = call_llm(system_prompt, user_prompt)
        except Exception as e:
            st.error(f"LLM call failed on batch {b + 1}: {e}")
            break

        # Strip accidental markdown fences
        cleaned = raw_response.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                all_raw.extend(parsed)
            elif isinstance(parsed, dict):
                all_raw.append(parsed)
        except json.JSONDecodeError as e:
            st.warning(f"Batch {b + 1}: could not parse LLM response as JSON ({e}).")
            with st.expander(f"Raw output — batch {b + 1}"):
                st.code(cleaned[:3000], language="text")

        progress.progress((b + 1) / n_batches)

    status.empty()
    progress.empty()

    valid_examples, validation_warnings = validate_and_convert(all_raw, config)

    st.session_state["gen_cache"] = {
        "raw_list":    all_raw,
        "valid":       valid_examples,
        "warnings":    validation_warnings,
        "config":      config,
    }

# ---- Results (rendered from cache, survives reruns) ----
if st.session_state.get("gen_cache"):
    cache   = st.session_state["gen_cache"]
    raw     = cache["raw_list"]
    valid   = cache["valid"]
    warns   = cache["warnings"]

    st.divider()
    st.subheader("Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Valid Examples",   len(valid))
    m2.metric("Warnings",         len(warns))
    m3.metric("Total Parsed",     len(raw))

    if warns:
        with st.expander(f"Validation warnings ({len(warns)})", expanded=False):
            for w in warns:
                st.write(f"- {w}")

    if valid:
        jsonl_bytes = to_jsonl(valid)

        st.download_button(
            label=f"Download JSONL ({len(valid)} examples)",
            data=jsonl_bytes,
            file_name="gliner2_training_data.jsonl",
            mime="application/jsonlines",
            key="download_btn",
        )

        with st.expander("Preview", expanded=False):
            for ex in valid:
                st.json(ex)

        with st.expander("Raw LLM output", expanded=False):
            st.code(
                json.dumps(raw, indent=2, ensure_ascii=False)[:6000],
                language="json",
            )
    else:
        st.error(
            "No valid examples were produced. "
            "Check the warnings above, adjust your schema or domain description, and try again."
        )
