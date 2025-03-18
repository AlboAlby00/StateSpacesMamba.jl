using Flux

struct ClassifierHeadDual
    dense1::Dense
    dense2::Dense
    dense3::Dense
end

Flux.@layer ClassifierHeadDual trainable = (dense1, dense2, dense3)

function ClassifierHeadDual(embed_dim, mlp_dim, num_classes)
    return ClassifierHeadDual(
        Dense(embed_dim * 2, mlp_dim, relu),
        Dense(mlp_dim, mlp_dim ÷ 2, relu),
        Dense(mlp_dim ÷ 2, num_classes),
    )
end

function (m::ClassifierHeadDual)(x1, x2; drop_degenerate_dim=true)
    x1, x2 = x1[:,end,:], x2[:,end,:]
    x = cat(x1, x2, dims=1)
    x = m.dense1(x)
    x = m.dense2(x)
    x = m.dense3(x)
    if size(x, 1) == 1 && drop_degenerate_dim
        x = dropdims(x, dims=1)
    end
    return x
end