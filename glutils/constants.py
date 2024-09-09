import jax.numpy as jnp

U1_GEN = 2j * jnp.ones((1, 1, 1))

SU2_GEN = 1j * jnp.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
])

SU3_GEN  = 1j * jnp.array([
    [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
    [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
    [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
    [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
    [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
    [[1/jnp.sqrt(3), 0, 0],
     [0, 1/jnp.sqrt(3), 0],
     [0, 0, -2/jnp.sqrt(3)]],
])
