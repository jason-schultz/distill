import Config

# Use the simpler BinaryBackend for tests to avoid Torchx loading issues
config :nx, default_backend: Nx.BinaryBackend
