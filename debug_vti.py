import base64, numpy as np, re, sys, zlib
from pathlib import Path

p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/arm.vti')
if not p.exists():
    raise FileNotFoundError(f"VTI file not found: {p}")

print(f"Inspecting: {p}")
text = p.read_text(errors='ignore')

match = re.search(r"<AppendedData[^>]*>\s*_(.*?)\s*</AppendedData>", text, re.S)
raw_b64 = ''.join(match.group(1).split())
print(f"Appended base64 length: {len(raw_b64)}")

eq_positions = [i for i, c in enumerate(raw_b64) if c == '=']
print(f"Number of = chars: {len(eq_positions)}")
print(f"= positions (first 10): {eq_positions[:10]}")

# decode first block (up to and including first run of '=')
p1_end = eq_positions[0]
while p1_end+1 < len(raw_b64) and raw_b64[p1_end+1] == '=':
    p1_end += 1
chunk1 = base64.b64decode(raw_b64[:p1_end+1])
print(f"\nChunk1 size: {len(chunk1)} bytes: {chunk1.hex()}")
print(f"As UInt32 LE: {np.frombuffer(chunk1, dtype='<u4')}")

# decode second block
p2_start = p1_end + 1
p2_end = None
for pos in eq_positions:
    if pos > p1_end:
        p2_end = pos
        while p2_end+1 < len(raw_b64) and raw_b64[p2_end+1] == '=':
            p2_end += 1
        break

if p2_end is not None:
    chunk2 = base64.b64decode(raw_b64[p2_start:p2_end+1])
    print(f"\nChunk2 size: {len(chunk2)} bytes")
    print(f"First 8 bytes: {chunk2[:8].hex()}")
    try:
        d = zlib.decompress(chunk2)
        print(f"Chunk2 decompressed: {len(d)} bytes")
    except Exception as e:
        print(f"Chunk2 zlib error: {e}")

    # third block
    chunk3_b64 = raw_b64[p2_end+1:]
    print(f"\nChunk3 b64 length: {len(chunk3_b64)}")
    try:
        chunk3 = base64.b64decode(chunk3_b64)
    except Exception:
        chunk3 = base64.b64decode(chunk3_b64 + '==')
    print(f"Chunk3 size: {len(chunk3)} bytes")
    print(f"First 24 bytes: {chunk3[:24].hex()}")
    if len(chunk3) >= 16:
        print(f"As UInt32 LE: {np.frombuffer(chunk3[:16], dtype='<u4')}")
else:
    print("Only one = group found")
    chunk23_b64 = raw_b64[p1_end+1:]
    chunk23 = base64.b64decode(chunk23_b64)
    print(f"Rest size: {len(chunk23)} bytes")
    print(f"First 32 bytes: {chunk23[:32].hex()}")
