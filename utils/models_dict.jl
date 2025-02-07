include("../models/mamba.jl")
include("../models/transformer.jl")

models = Dict(
    "transformer" => TransformerGPT(alphabet, seq_len, n_embed=512, n_layers=6), # 1987139 parameters
    "transformer_with_dropout" => TransformerGPT(alphabet, seq_len, n_embed=512, n_layers=6, dropout=0.1), 
    "mamba" => MambaGPT(vocab_size, embed_dim = 128, N = 16, n_layers=6), # 1927859 parameters
    "mamba_with_dropout" => MambaGPT(vocab_size, embed_dim = 128, N = 16, n_layers=6, dropout=0.1),
    "transformer_small" => TransformerGPT(alphabet, seq_len, n_embed=256, n_layers=3),
    "mamba_small" => MambaGPT(vocab_size, embed_dim = 128, N = 8, n_layers=3),
    "mamba_small_with_dropout" => MambaGPT(vocab_size, embed_dim = 128, N = 8, n_layers=3, dropout=0.1),
    "bayesian_mamba" => MambaGPT(vocab_size, embed_dim = 128, N = 8, n_layers=3)
)