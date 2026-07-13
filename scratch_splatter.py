import re

with open("wasm_frontend/src/adapters/cpu_splatter.rs", "r") as f:
    code = f.read()

# 1. Update shade_and_splat signature
code = code.replace(
    "fn shade_and_splat(\n        &mut self,\n        cam: &Camera,\n        center: [f32; 3],\n",
    "fn shade_and_splat(\n        &mut self,\n        cam: &Camera,\n        chunks: &[ChunkDraw],\n        center: [f32; 3],\n        world_size: f32,\n"
)

# 2. Update all calls inside render_node
code = re.sub(
    r"self\.shade_and_splat\(\n(\s+)cam,\n(\s+)center,",
    r"self.shade_and_splat(\n\1cam,\n\1chunks,\n\1center,\n\1world_size,",
    code
)

# 3. Update tests
code = re.sub(
    r"r\.shade_and_splat\(\n(\s+)&cam,\n(\s+)center,",
    r"r.shade_and_splat(\n\1&cam,\n\1&chunks,\n\1center,\n\1world_size,",
    code
)

# Wait, in tests, chunks is defined?
# Let's check tests
# We can just write out the new code and see what breaks in cargo check.

with open("wasm_frontend/src/adapters/cpu_splatter.rs", "w") as f:
    f.write(code)
