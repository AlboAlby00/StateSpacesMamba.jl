using Flux, Optimisers, ProgressMeter
using Statistics
using MLDatasets
using CUDA
CUDA.allowscalar(false)

include("models/mamba_layer.jl")
using Main.MambaLayer

# hyperparameters

batch_size = 16
test_batch_size = 8
hidden_dim = 128 # D in the paper
mamba_block_dim = 16 # N in the paper
epochs = 1000
device = gpu_device()

n_classes = 10
input_dim = 28 *14
mambaArgs = MambaLayer.MambaArgs(input_dim, n_classes, D=hidden_dim, N=mamba_block_dim)

# create split mnist dataloader

# train loader
train_images, train_labels = MNIST(split=:train)[:]
n_train_images = size(train_images)[end]
train_sequences = reshape(train_images, input_dim, 28 * 28 ÷ input_dim, n_train_images) # shape is (d, l, b) d -> feature dimension, l -> sequence length, b -> batch size
# println(train_labels[1:10]) # 60000
train_labels = repeat(train_labels', outer=(28 * 28 ÷ input_dim, 1))
# println(train_labels[1:end, 1:10]) # 4, 600000
train_loader = Flux.DataLoader((train_sequences, train_labels), batchsize=batch_size, shuffle=true, partial=false)

for (x, y) in train_loader
    @assert size(x) == (input_dim, 28 * 28 / input_dim, batch_size)
end

# test loader
test_images, test_labels = MNIST(split=:test)[:]
n_test_images = size(test_images)[end]
test_sequences = reshape(test_images, input_dim, 28 * 28 ÷ input_dim, n_test_images)
test_labels = repeat(test_labels', outer=(28 * 28 ÷ input_dim, 1))
test_loader = Flux.DataLoader((test_sequences, test_labels), batchsize=test_batch_size, partial=false)

for (x, y) in test_loader
    @assert size(x) == (input_dim, 28 * 28 / input_dim, test_batch_size)
end

mamba_model = MambaLayer.Mamba(mambaArgs) |> device |> f64

fc_model = Chain(
               x -> reshape(x, 28 * 28, size(x, 3)),
               Dense(28 * 28, 128, relu),
               Dense(128, 10),
           ) |> device |> f64

model = mamba_model

opt_state = Flux.setup(Flux.Adam(0.001), model)

losses = []
@showprogress "Training progress..." for epoch in 1:epochs # Training loop
    Flux.reset!(model)

    # Validation loop
    accuracy_list = []
    test_progress = Progress(length(test_loader))
    count = 0
    last_predictions = []
    last_y = []
    last_predictions_old = []
    for (x, y) in test_loader
        x = x |> device
        logits = Array(model(x)) # move back to CPU # 1, l, b
        predictions = dropdims(argmax(logits, dims=1), dims=1) # l, b
        # convert CartesianIndex matrix to int matrix
        last_predictions_old = predictions
        temp = zeros(Int, size(predictions))
        for idx in eachindex(predictions)
            temp[idx] = Int(predictions[idx][1] - 1)  # Extract and convert to Int
        end
        accuracy = sum(temp .== y) / length(temp)
        push!(accuracy_list, accuracy)
        next!(test_progress; showvalues=[("Epoch ", epoch), ("Accuracy ", mean(accuracy_list))])
        last_predictions = temp
        last_y = y
    end
    # println(last_y[:,:])
    # println(last_predictions[:,:])
    # println(last_predictions_old[:,:])

    # Training loop
    train_progress = Progress(length(train_loader))
    for (x, y) in train_loader
        x, y = x |> device, y |> device
        loss, grads = Flux.withgradient(model) do m
            logits = m(x)
            onehot_y = Flux.onehotbatch(y, 0:n_classes-1)
            Flux.Losses.logitcrossentropy(logits, onehot_y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
        next!(train_progress; showvalues=[("Epoch ", epoch), ("Loss ", loss)])
    end
end
