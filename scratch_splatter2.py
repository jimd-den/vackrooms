with open("wasm_frontend/src/adapters/cpu_splatter.rs", "r") as f:
    code = f.read()

# Replace world_size with size ONLY in the render_node method body
# Actually, the error shows it's lines 858, 878, 923, 952.
# We can just replace "center,\n                        world_size," with "center,\n                        size,"

code = code.replace(
    "center,\n                        world_size,",
    "center,\n                        size,"
)
code = code.replace(
    "center,\n                    world_size,",
    "center,\n                    size,"
)

with open("wasm_frontend/src/adapters/cpu_splatter.rs", "w") as f:
    f.write(code)
