"""
GGUF Architecture Patcher — called by the Node backend as a subprocess.

Reads a GGUF file, changes general.architecture and renames all
architecture-prefixed metadata keys (e.g. qwen35.* → qwen3.*),
rewrites the metadata + tensor-info section, and streams tensor data
verbatim.  The output file is structurally valid GGUF v3.

Usage:
  python3 gguf_patch.py <input.gguf> <output.gguf> <new_arch>
"""

import struct, sys, os
import numpy as np

MAGIC = 0x46554747  # 'GGUF' LE

# ── Helpers ──────────────────────────────────────────────────────

class Buffer:
    def __init__(self, data: bytes):
        self.data = bytearray(data) if isinstance(data, bytes) else bytearray(data)
        self.pos = 0

    def u8(self):
        v = self.data[self.pos]; self.pos += 1; return v
    def i8(self):
        v = struct.unpack_from('<b', self.data, self.pos)[0]; self.pos += 1; return v
    def u16(self):
        v = struct.unpack_from('<H', self.data, self.pos)[0]; self.pos += 2; return v
    def i16(self):
        v = struct.unpack_from('<h', self.data, self.pos)[0]; self.pos += 2; return v
    def u32(self):
        v = struct.unpack_from('<I', self.data, self.pos)[0]; self.pos += 4; return v
    def i32(self):
        v = struct.unpack_from('<i', self.data, self.pos)[0]; self.pos += 4; return v
    def f32(self):
        v = struct.unpack_from('<f', self.data, self.pos)[0]; self.pos += 4; return v
    def u64(self):
        v = struct.unpack_from('<Q', self.data, self.pos)[0]; self.pos += 8; return v
    def i64(self):
        v = struct.unpack_from('<q', self.data, self.pos)[0]; self.pos += 8; return v
    def f64(self):
        v = struct.unpack_from('<d', self.data, self.pos)[0]; self.pos += 8; return v
    def bool_v(self):
        return self.u8() != 0
    def string(self):
        length = self.u64()
        s = self.data[self.pos:self.pos + length].decode('utf-8')
        self.pos += length
        return s

    def bytes_range(self, length):
        r = bytes(self.data[self.pos:self.pos + length])
        self.pos += length
        return r

    def skip_value(self, vtype: int):
        if vtype == 0 or vtype == 1: self.pos += 1       # u8/i8
        elif vtype == 2 or vtype == 3: self.pos += 2     # u16/i16
        elif vtype == 4 or vtype == 5 or vtype == 6: self.pos += 4  # u32/i32/f32
        elif vtype == 7: self.pos += 1                   # bool
        elif vtype == 8: self.string()                   # string
        elif vtype == 10 or vtype == 11 or vtype == 12: self.pos += 8  # u64/i64/f64
        elif vtype == 9:                                 # array
            elem_type = self.u32()
            count = self.u64()
            for _ in range(count):
                self.skip_value(elem_type)


# ── Writer helper ─────────────────────────────────────────────────

class Writer:
    def __init__(self):
        self.buf = bytearray()

    def write(self, data): self.buf.extend(data)
    def u8(self, v): self.write(struct.pack('<B', v))
    def i8(self, v): self.write(struct.pack('<b', v))
    def u16(self, v): self.write(struct.pack('<H', v))
    def i16(self, v): self.write(struct.pack('<h', v))
    def u32(self, v): self.write(struct.pack('<I', v))
    def i32(self, v): self.write(struct.pack('<i', v))
    def f32(self, v): self.write(struct.pack('<f', v))
    def u64(self, v): self.write(struct.pack('<Q', v))
    def i64(self, v): self.write(struct.pack('<q', v))
    def f64(self, v): self.write(struct.pack('<d', v))
    def bool_w(self, v): self.u8(1 if v else 0)

    def string(self, s: str):
        b = s.encode('utf-8')
        self.u64(len(b))
        self.write(b)

    def write_value(self, vtype: int, raw: memoryview | bytes | bytearray):
        """Write a GGUF value given its type and raw bytes from the source."""
        if vtype == 0: self.u8(raw[0])
        elif vtype == 1: self.i8(raw[0] if raw[0] < 128 else raw[0] - 256)
        elif vtype == 2: self.u16(struct.unpack_from('<H', raw, 0)[0])
        elif vtype == 3: self.i16(struct.unpack_from('<h', raw, 0)[0])
        elif vtype == 4: self.u32(struct.unpack_from('<I', raw, 0)[0])
        elif vtype == 5: self.i32(struct.unpack_from('<i', raw, 0)[0])
        elif vtype == 6: self.f32(struct.unpack_from('<f', raw, 0)[0])
        elif vtype == 7: self.bool_w(raw[0] != 0)
        elif vtype == 8:  # string: raw already includes length prefix?
            # In our metadata capture, raw for STRING type contains just
            # the value bytes, not the length prefix.
            length = len(raw)
            self.u64(length)
            self.write(raw)
        elif vtype == 10: self.u64(struct.unpack_from('<Q', raw, 0)[0])
        elif vtype == 11: self.i64(struct.unpack_from('<q', raw, 0)[0])
        elif vtype == 12: self.f64(struct.unpack_from('<d', raw, 0)[0])
        elif vtype == 9:  # array
            self.write(raw)
        else:
            self.write(raw)


# ── Tensor info writer ────────────────────────────────────────────

class TensorEntry:
    __slots__ = ('name', 'n_dims', 'dims', 'tensor_type', 'offset')
    def __init__(self, name, shape, tensor_type, offset):
        self.name = name
        self.n_dims = len(shape)
        self.dims = shape
        self.tensor_type = int(tensor_type)
        self.offset = offset


# ── Main ──────────────────────────────────────────────────────────

def parse_gguf(filepath: str):
    """Parse GGUF metadata. Returns (kv_entries, tensor_entries, data_start, alignment)."""
    with open(filepath, 'rb') as f:
        data = f.read()

    buf = Buffer(data)
    magic = buf.u32()
    if magic != MAGIC:
        raise ValueError(f"Not a GGUF file (magic={hex(magic)})")

    version = buf.u32()
    tensor_count = buf.u64()
    kv_count = buf.u64()

    # Read KV pairs
    kvs = []
    alignment = 32
    old_arch = None

    for _ in range(kv_count):
        key_start = buf.pos
        key = buf.string()
        vtype = buf.u32()

        # Capture the raw bytes of the value
        val_start = buf.pos
        if vtype == 8:  # STRING
            slen = buf.u64()
            raw = buf.bytes_range(slen)
            kvs.append((key, vtype, slen, raw))
        elif vtype in (0, 1):  # u8/i8
            kvs.append((key, vtype, 1, data[val_start:val_start+1]))
            buf.pos += 1
        elif vtype in (2, 3):  # u16/i16
            kvs.append((key, vtype, 2, data[val_start:val_start+2]))
            buf.pos += 2
        elif vtype in (4, 5, 6):  # u32/i32/f32
            kvs.append((key, vtype, 4, data[val_start:val_start+4]))
            buf.pos += 4
        elif vtype == 7:  # BOOL
            kvs.append((key, vtype, 1, data[val_start:val_start+1]))
            buf.pos += 1
        elif vtype in (10, 11, 12):  # u64/i64/f64
            kvs.append((key, vtype, 8, data[val_start:val_start+8]))
            buf.pos += 8
        elif vtype == 9:  # ARRAY
            elem_type = buf.u32()
            arr_len = buf.u64()
            arr_start = val_start
            for _ in range(arr_len):
                if elem_type == 8:
                    s_len = buf.u64()
                    buf.pos += s_len
                elif elem_type in (0, 1): buf.pos += 1
                elif elem_type in (2, 3): buf.pos += 2
                elif elem_type in (4, 5, 6): buf.pos += 4
                elif elem_type == 7: buf.pos += 1
                elif elem_type in (10, 11, 12): buf.pos += 8
            arr_end = buf.pos
            kvs.append((key, vtype, arr_end - arr_start, data[arr_start:arr_end]))
        else:
            kvs.append((key, vtype, 0, b''))

        if key == 'general.architecture':
            _, _, slen, raw = kvs[-1]
            old_arch = raw.decode('utf-8')
        if key == 'general.alignment' and vtype in (4, 5):
            alignment = struct.unpack_from('<I', raw)[0]

    # Read tensor info entries
    tensor_entries = []
    for _ in range(tensor_count):
        name = buf.string()
        n_dims = buf.u32()
        dims = [buf.u64() for _ in range(n_dims)]
        tensor_type = buf.u32()
        offset = buf.u64()
        tensor_entries.append(TensorEntry(name, tuple(dims), tensor_type, offset))

    tensor_info_end = buf.pos
    data_start = ((tensor_info_end + alignment - 1) // alignment) * alignment

    return kvs, tensor_entries, data_start, alignment, filepath, old_arch


def write_gguf(out_path, kvs, tensor_entries, data_start, version,
               old_arch, new_arch, src_path):
    """Write a new GGUF file with the architecture renamed."""

    w = Writer()

    # Header
    w.u32(MAGIC)
    w.u32(version)
    w.u64(len(tensor_entries))
    w.u64(len(kvs))

    # Rename KV entries
    for key, vtype, raw_len, raw in kvs:
        new_key = key
        new_raw = raw
        if key == 'general.architecture':
            new_raw = new_arch.encode('utf-8')
        elif key == 'tokenizer.ggml.pre' and old_arch:
            new_raw = raw.replace(old_arch.encode('utf-8'),
                                  new_arch.encode('utf-8'))
        elif old_arch and key.startswith(old_arch + '.'):
            new_key = new_arch + key[len(old_arch):]

        # Write key
        w.string(new_key)
        # Write value type
        w.u32(vtype)

        if key == 'general.architecture':
            w.string(new_arch)
        elif key == 'tokenizer.ggml.pre' and old_arch:
            w.string(new_raw.decode('utf-8'))
        else:
            w.write_value(vtype, raw)

    # Tensor info
    for t in tensor_entries:
        w.string(t.name)
        w.u32(t.n_dims)
        for d in t.dims:
            w.u64(d)
        w.u32(t.tensor_type)
        w.u64(t.offset)

    # Padding to alignment
    alignment = 32
    for _, vtype, _, raw in kvs:
        if _ == 'general.alignment' and vtype in (4, 5):
            alignment = struct.unpack_from('<I', raw)[0]
    pad = (alignment - (len(w.buf) % alignment)) % alignment
    if pad:
        w.write(b'\x00' * pad)

    new_data_start = len(w.buf)
    delta = new_data_start - data_start

    # Adjust tensor offsets in the written data
    if delta != 0:
        # We need to update the tensor offsets that were already written.
        # They're relative to data_start, so they'd be wrong if data_start changed.
        # Since we re-wrote metadata from scratch, data_start may change.
        # For now, we log this and trust that relative offsets are maintained
        # because tensor offsets are relative to the data section.
        pass

    # Write header to file
    with open(out_path, 'wb') as outf:
        outf.write(w.buf)

        # Copy tensor data from source
        with open(src_path, 'rb') as inf:
            inf.seek(data_start)
            chunk = 16 * 1024 * 1024
            copied = 0
            total = os.path.getsize(src_path) - data_start
            while True:
                data_chunk = inf.read(chunk)
                if not data_chunk:
                    break
                outf.write(data_chunk)
                copied += len(data_chunk)
                # Progress for the calling process
                pct = int(copied * 100 / total) if total else 100
                print(f"PROGRESS:{copied}:{total}", flush=True)

    return new_data_start


# ── Architecture suggestion ───────────────────────────────────────

def suggest_target(arch: str) -> str | None:
    a = arch.lower()
    if a == 'qwen35' or a == 'qwen3.5' or a == 'qwen36' or a == 'qwen3.6':
        return 'qwen3'
    if a == 'qwen25' or a == 'qwen2.5':
        return 'qwen2'
    if a == 'mixtral':
        return 'llama'
    return None


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <input.gguf> <output.gguf> <new_arch>")
        sys.exit(1)

    src = sys.argv[1]
    dst = sys.argv[2]
    new_arch = sys.argv[3]

    if not os.path.exists(src):
        print(f"ERROR: Source not found: {src}", flush=True)
        sys.exit(1)

    os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)

    kvs, tensors, data_start, alignment, src_path, old_arch = parse_gguf(src)

    print(f"INFO: old_arch={old_arch} new_arch={new_arch}", flush=True)
    print(f"INFO: version=3 tensors={len(tensors)} kvs={len(kvs)}", flush=True)
    print(f"INFO: data_start={data_start}", flush=True)

    new_data_start = write_gguf(dst, kvs, tensors, data_start, 3,
                                old_arch, new_arch, src_path)

    out_size = os.path.getsize(dst)
    print(f"COMPLETE:{out_size}", flush=True)
