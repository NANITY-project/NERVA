"""Write a tiny random-weights NANITY-spec GGUF (v1 spec). It generates gibberish, but it is a *valid* model:
the real NEON binary loads it, passes --probe, and streams tokens, which lets the integration be tested end to end.

    python3 make_tiny_gguf.py out.gguf [--arch nanity|llama] [--spec 1]
"""
import random
import struct
import sys

def build(path, arch="nanity", spec=1, seed=0, n_embd=32, n_layer=2, n_head=4, n_kv=2, head_dim=8, n_ff=64, ctx=512):
    rnd = random.Random(seed)
    words = ["GRAB", "WAIT", "SAY", "WALK_TO", "LOOK_AT", " right", " left", " cup", " bob", " 1", "\n"]
    tokens = ["<unk>", "<s>", "</s>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"] + [f"<0x{b:02X}>" for b in range(256)] + words
    n_vocab = len(tokens)
    q_dim, kv_dim = n_head * head_dim, n_kv * head_dim

    def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
    kvs = []
    def kv_str(k, v): kvs.append(s(k) + struct.pack("<I", 8) + s(v))
    def kv_u32(k, v): kvs.append(s(k) + struct.pack("<II", 4, v))
    def kv_f32(k, v): kvs.append(s(k) + struct.pack("<If", 6, v))
    kv_str("general.architecture", arch); kv_str("general.name", f"tiny-random-{arch}"); kv_u32("general.alignment", 32)
    kv_u32("general.file_type", 0)
    kv_u32(f"{arch}.spec_version", spec); kv_u32(f"{arch}.embedding_length", n_embd); kv_u32(f"{arch}.block_count", n_layer)
    kv_u32(f"{arch}.attention.head_count", n_head); kv_u32(f"{arch}.attention.head_count_kv", n_kv)
    kv_u32(f"{arch}.attention.key_length", head_dim); kv_u32(f"{arch}.feed_forward_length", n_ff); kv_u32(f"{arch}.context_length", ctx)
    kv_f32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5); kv_f32(f"{arch}.rope.freq_base", 10000.0)
    kvs.append(s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, n_vocab) + b"".join(s(t) for t in tokens))
    kv_u32("tokenizer.ggml.bos_token_id", 1); kv_u32("tokenizer.ggml.eos_token_id", 2); kv_u32("tokenizer.ggml.unknown_token_id", 0)

    tensors = [("token_embd.weight", [n_embd, n_vocab], "rand"), ("output_norm.weight", [n_embd], "ones")]
    for i in range(n_layer):
        tensors += [(f"blk.{i}.attn_norm.weight", [n_embd], "ones"), (f"blk.{i}.attn_q.weight", [n_embd, q_dim], "rand"),
                    (f"blk.{i}.attn_k.weight", [n_embd, kv_dim], "rand"), (f"blk.{i}.attn_v.weight", [n_embd, kv_dim], "rand"),
                    (f"blk.{i}.attn_output.weight", [q_dim, n_embd], "rand"), (f"blk.{i}.ffn_norm.weight", [n_embd], "ones"),
                    (f"blk.{i}.ffn_gate.weight", [n_embd, n_ff], "rand"), (f"blk.{i}.ffn_up.weight", [n_embd, n_ff], "rand"),
                    (f"blk.{i}.ffn_down.weight", [n_ff, n_embd], "rand")]
    infos, data, off = [], b"", 0
    for name, ne, kind in tensors:
        n = 1
        for d in ne: n *= d
        raw = struct.pack(f"<{n}f", *([1.0] * n if kind == "ones" else [rnd.gauss(0, 0.05) for _ in range(n)]))
        infos.append(s(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne) + struct.pack("<IQ", 0, off))
        pad = (-len(raw)) % 32
        data += raw + b"\0" * pad; off += len(raw) + pad
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(infos)
    head += b"\0" * ((-len(head)) % 32)
    with open(path, "wb") as f:
        f.write(head + data)

if __name__ == "__main__":
    args = sys.argv[1:]
    arch = args[args.index("--arch") + 1] if "--arch" in args else "nanity"
    spec = int(args[args.index("--spec") + 1]) if "--spec" in args else 1
    build(args[0], arch, spec)
    print("wrote", args[0])
