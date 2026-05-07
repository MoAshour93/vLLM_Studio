"""
Sets up transformers GGUF support for community arch labels.
Called once at backend startup. Idempotent.
"""
import transformers.integrations.ggml as g
import transformers.modeling_gguf_pytorch_utils as m

COMMUNITY = {
    'qwen35': 'qwen3',
    'qwen36': 'qwen3',
    'qwen25': 'qwen2',
    'qwen35moe': 'qwen3_moe',
}

MODEL_TYPE_REMAPS = {
    'qwen35': 'qwen3_5',
    'qwen35moe': 'qwen3_5_moe',
}

changed = False

# Patch GGUF_CONFIG_MAPPING (runtime + disk)
for label, official in COMMUNITY.items():
    if label not in g.GGUF_CONFIG_MAPPING:
        g.GGUF_CONFIG_MAPPING[label] = g.GGUF_CONFIG_MAPPING[official].copy()
        changed = True

# Persist to disk
if changed:
    with open(g.__file__, 'r') as f:
        lines = f.readlines()
    # Insert after the closing brace of GGUF_CONFIG_MAPPING (line ~323)
    for i, line in enumerate(lines):
        if line.strip() == '}' and i > 300 and 'TOKENIZER' in (lines[i+1] if i+1 < len(lines) else ''):
            for label in COMMUNITY:
                if f'["{label}"]' not in ''.join(lines):
                    lines.insert(i, f'GGUF_CONFIG_MAPPING["{label}"] = GGUF_CONFIG_MAPPING["{COMMUNITY[label]}"].copy()\n')
            with open(g.__file__, 'w') as f:
                f.writelines(lines)
            break

# Patch model_type remap (disk)
with open(m.__file__, 'r') as f:
    r = f.read()

rt = 'model_type = hf_model.config.model_type if model_type is None else model_type'
for label, mtype in MODEL_TYPE_REMAPS.items():
    if f'model_type == "{label}"' not in r:
        r = r.replace(rt, rt + f'\n    if model_type == "{label}":\n        model_type = "{mtype}"')
        changed = True

if changed or 'model_type == "qwen35"' in r:
    with open(m.__file__, 'w') as f:
        f.write(r)

# Also add parsed_parameters model_type remap (fixes qwen35 CONFIG_MAPPING lookup)
old_pp = 'if parsed_parameters["config"]["model_type"] == "gemma3":\n        parsed_parameters["config"]["model_type"] = "gemma3_text"'
new_pp = 'if parsed_parameters["config"].get("model_type") == "qwen35":\n        parsed_parameters["config"]["model_type"] = "qwen3_5"\n    ' + old_pp
if 'parsed_parameters["config"].get("model_type") == "qwen35"' not in r:
    r = r.replace(old_pp, new_pp)
    with open(m.__file__, 'w') as f:
        f.write(r)
    changed = True

print(f"GGUF setup {'patched' if changed else 'already configured'}")
