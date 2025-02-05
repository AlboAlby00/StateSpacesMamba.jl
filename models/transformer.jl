struct GPTBlock
    layernorm1::LayerNorm
    mha::MultiHeadAttention
    mlp::Chain
end

Flux.@layer GPTBlock

function GPTBlock(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout)
    GPTBlock(
        LayerNorm(n_embed),
        MultiHeadAttention(n_embed => (qk_dim, v_dim) => n_embed; nheads=n_heads, dropout_prob=dropout),
        Chain(
            LayerNorm(n_embed),
            Dense(n_embed => n_hidden, gelu),
            Dense(n_hidden => n_embed),
            Dropout(dropout)
        ),
    )
end

function (m::GPTBlock)(x)
    y, α = m.mha(m.layernorm1(x); mask=NNlib.make_causal_mask(x))
    x += y
    x += m.mlp(x)
    return x
end

struct TransformerGPT
    alphabet::Vector{Char}
    tok_embed::Embedding
    pos_embed::Embedding
    dropout::Dropout
    blocks::Vector{GPTBlock}
    layernorm1::LayerNorm
    output_layer::Dense
end

Flux.@layer TransformerGPT

function TransformerGPT(alphabet::AbstractVector{Char}, seq_len; n_embed=64, n_hidden=256, qk_dim=16, v_dim=16, n_heads=4, dropout=0.0)
    n_vocab = length(alphabet)
    TransformerGPT(
        alphabet,
        Embedding(n_vocab => n_embed),
        Embedding(seq_len => n_embed),
        Dropout(dropout),
        map(_ -> GPTBlock(
            n_embed  = n_embed,
            n_hidden = n_hidden,
            qk_dim   = qk_dim,
            v_dim    = v_dim,
            n_heads  = n_heads,
            dropout  = dropout), 1:n_layers),
        LayerNorm(n_embed),
        Dense(n_embed => n_vocab),
    )
end

function (m::TransformerGPT)(tokens)
    T, B = size(tokens)
    te = m.tok_embed(tokens)
    pe = m.pos_embed(1:T)
    x = m.dropout(te .+ pe)
    for blk in m.blocks
        x = blk(x)
    end
    x = m.layernorm1(x)
    x = m.output_layer(x)
    return x
end