# ABOUTME: In tên và shape các cổng vào/ra của file ONNX
# ABOUTME: Dùng để điền config.pbtxt cho đúng - sai tên là Triton không load

import sys

import onnx


def dims(t):
    return [d.dim_value or d.dim_param or "?" for d in t.type.tensor_type.shape.dim]


for path in sys.argv[1:]:
    m = onnx.load(path)
    print(f"\n=== {path} ===")
    for i in m.graph.input:
        print(f"  IN   {i.name:20s} {dims(i)}")
    for o in m.graph.output:
        print(f"  OUT  {o.name:20s} {dims(o)}")
    if m.metadata_props:
        print("  META")
        for p in m.metadata_props:
            print(f"       {p.key} = {p.value}")
