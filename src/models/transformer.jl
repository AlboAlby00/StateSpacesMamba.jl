using Flux

include("classifier_head_dual.jl")

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

struct TransformerEncoder
    alphabet::Vector{Char}
    tok_embed::Embedding
    pos_embed::Embedding
    dropout::Dropout
    blocks::Vector{GPTBlock}
end

Flux.@layer TransformerEncoder

function TransformerEncoder(alphabet::AbstractVector{Char}, seq_len, n_embed, n_hidden,  n_layers, qk_dim, v_dim, n_heads, dropout)
    n_vocab = length(alphabet)
    TransformerEncoder(
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
            dropout  = dropout), 1:n_layers)
    )
end

function (m::TransformerEncoder)(tokens)
    T, B = size(tokens)
    te = m.tok_embed(tokens)
    pe = m.pos_embed(1:T)
    x = m.dropout(te .+ pe)
    for blk in m.blocks
        x = blk(x)
    end
    return x
end

function TransformerGPT(alphabet::AbstractVector{Char}, seq_len; n_embed=64, n_hidden=256,  n_layers=3, qk_dim=16, v_dim=16, n_heads=4, dropout=0.0)
    n_vocab = length(alphabet)
    model = Chain(
        TransformerEncoder(alphabet, seq_len, n_embed, n_hidden, n_layers, qk_dim, v_dim, n_heads, dropout),
        Dense(n_embed => n_vocab)
    )
    return model
end

# Define the Transformer dual encoder for the LRA Document Retrieval task

struct TransformerDualEncoder
    shared_encoder::TransformerEncoder
    classifier_head::ClassifierHeadDual
    dropout::Dropout
end
Flux.@layer TransformerDualEncoder trainable = (shared_encoder, classifier_head)

function TransformerDualEncoder(alphabet::AbstractVector{Char}, seq_len; n_embed=64, n_hidden=256,  n_layers=3, qk_dim=16, v_dim=16, n_heads=4, dropout=0.0)

    shared_encoder = TransformerEncoder(alphabet, seq_len, n_embed, n_hidden, n_layers, qk_dim, v_dim, n_heads, dropout)
    classifier_head = ClassifierHeadDual(n_embed, n_embed, 1)
    dropout_layer = Dropout(dropout)

    return TransformerDualEncoder(shared_encoder, classifier_head, dropout_layer)
end

function (m::TransformerDualEncoder)(x1, x2)

    out1 = m.shared_encoder(x1)
    out2 = m.shared_encoder(x2)
    out = m.classifier_head(out1, out2)

    return out
end