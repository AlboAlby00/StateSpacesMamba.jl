using Flux, Optimisers, ProgressMeter
using Statistics
using MLDatasets
using CUDA
using Revise
using Plots
using DelimitedFiles

CUDA.allowscalar(false)

include("models/mamba.jl")
include("models/transformer.jl")
include("parser.jl")
include("utils/shakespeare.jl")
include("utils/common.jl")
using Main.MambaLayer

device = gpu_device()

train_batch_size = 128
test_batch_size = 64
embed_dim = 256
model_name = "mamba"
seq_len = 90
save_csv = true

alphabet, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len=seq_len)
vocab_size = length(alphabet)

train_loader = Flux.DataLoader((trainX, trainY), batchsize=train_batch_size, shuffle=true, partial=false) |> device
test_loader = Flux.DataLoader((testX, testY), batchsize=test_batch_size, shuffle=false, partial=false) |> device

vocab_size = length(alphabet)

models = Dict(
    "transformer" => TransformerGPT(alphabet, seq_len, n_embed=embed_dim),
    "mamba" => MambaLayer.MambaGPT(vocab_size, embed_dim = embed_dim, N = 16, n_layers=3 )
)
model = models[model_name] |> f32 |> device

for (name, model) in models
    println("Number of parameters of model $name is $(count_params(model))")
end

opt = Optimisers.Adam(0.001)
opt_state = Optimisers.setup(opt, model)

function criterion(logits, y)
    # y is one hot encoded
    Flux.Losses.logitcrossentropy(logits, y)
end

num_epochs = 10
@showprogress for epoch in 1:num_epochs

    # Validation loop
    Flux.testmode!(model)
    test_progress = Progress(length(test_loader))
    losses = []
    for (x, y) in test_loader
        y = y |> cpu
        logits = model(x) |> cpu

        test_loss = criterion(logits, y)

        push!(losses, test_loss)

        next!(test_progress; showvalues=[("Epoch ", epoch), ("Test Loss ", mean(losses))])
    end
    push!(test_losses, mean(losses))

    println(generate(model, alphabet, "Before", 50, embed_dim))

    # Training loop
    Flux.trainmode!(model)
    loss_moving_avg = nothing
    α = 0.05
    train_progress = Progress(length(train_loader))
    for (x, y) in train_loader

        loss, grads = Flux.withgradient(m -> criterion(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])
        # Update moving average of the loss
        if loss_moving_avg === nothing
            loss_moving_avg = loss
        else
            loss_moving_avg = α * loss + (1 - α) * loss_moving_avg
        end
        push!(train_losses, loss)

        next!(train_progress; showvalues=[("Epoch ", epoch), ("Training Loss ", loss_moving_avg)])
    end

end

if save_csv
    writedlm( "saved_csv/train_losses_$model_name.csv",  train_losses, ',')
    writedlm( "saved_csv/test_losses_$model_name.csv",  test_losses, ',')
end

