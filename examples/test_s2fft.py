import sys
print("1. Starting script...", flush=True)

import os
# Uncomment this if you want to force CUDA
os.environ["JAX_PLATFORMS"] = "cuda"

print("2. Importing JAX...", flush=True)
import jax

print(f"3. JAX imported. Devices available: {jax.devices()}", flush=True)

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

print("4. Importing s2fft...", flush=True)
import s2fft

print("5. Creating array...", flush=True)
m = jnp.ones((17, 32))  # MWSS shape for L=16 is (L+1, 2L)
print(f"Original shape: {m.shape}", flush=True)

print("6. Running s2fft forward transform...", flush=True)
flm = s2fft.transforms.spherical.forward(
    m, L=16, reality=True, method="jax", sampling="mwss"
)
print(f"L=16 works! Output shape: {flm.shape}", flush=True)

# import sys
# print("1. Starting script...", flush=True)

# import jax
# # Force CPU mode programmatically to bypass the GPU XLA compilation crash
# jax.config.update("jax_platform_name", "cpu")
# jax.config.update("jax_enable_x64", True)

# print(f"3. JAX devices available: {jax.devices()}", flush=True)

# import jax.numpy as jnp
# import s2fft

# print("4. Importing s2fft...", flush=True)
# m = jnp.ones((17, 32))  # MWSS shape for L=16 is (L+1, 2L)
# print(f"Original shape: {m.shape}", flush=True)

# print("6. Running s2fft forward transform...", flush=True)
# flm = s2fft.transforms.spherical.forward(
#     m, L=16, reality=True, method="jax", sampling="mwss"
# )
# print(f"L=16 works! Output shape: {flm.shape}", flush=True)