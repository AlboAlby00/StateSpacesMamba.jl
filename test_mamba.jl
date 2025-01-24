using Flux, Optimisers, ProgressMeter
using Statistics
using MLDatasets
using CUDA
CUDA.allowscalar(false)

include("models/mamba_layer.jl")
include("parser.jl")
using Main.MambaLayer

# hyperparameters

train_batch_size = 32
test_batch_size = 32
epochs = 1000
device = gpu

n_classes = 10
input_dim = 1
args = MambaLayer.MambaArgs(input_dim, n_classes, mamba_block_output_dim=64, expand=128, N=8, n_layers=3)
# create split mnist dataloader

# train loader
train_images, train_labels = MNIST(split=:train)[:]
n_train_images = size(train_images)[end]
train_sequences = Float64.(reshape(train_images, input_dim, 28 * 28 ÷ input_dim, n_train_images)) # shape is (d, l, b) d -> feature dimension, l -> sequence length, b -> batch size
train_labels = repeat(train_labels', outer=(28 * 28 ÷ input_dim, 1))
train_loader = Flux.DataLoader((train_sequences, train_labels), batchsize=train_batch_size, shuffle=true, partial=false)

for (x, y) in train_loader
    @assert size(x) == (input_dim, 28 * 28 / input_dim, train_batch_size)
end

# test loader
test_images, test_labels = MNIST(split=:test)[:]
n_test_images = size(test_images)[end]
test_sequences = Float64.(reshape(test_images, input_dim, 28 * 28 ÷ input_dim, n_test_images))
test_labels = repeat(test_labels', outer=(28 * 28 ÷ input_dim, 1))
test_loader = Flux.DataLoader((test_sequences, test_labels), batchsize=test_batch_size, partial=false)

for (x, y) in test_loader
    @assert size(x) == (input_dim, 28 * 28 / input_dim, test_batch_size)
end

model = Chain(
            MambaLayer.Mamba(args)
        ) |> device |> f64

opt_state = Flux.setup(Flux.Adam(0.001), model)

@showprogress "Training progress..." for epoch in 1:epochs # Training loop
    # Flux.reset!(model)

    # Validation loop
    accuracy_list = []
    test_progress = Progress(length(test_loader))
    for (x, y) in test_loader
        x = x |> device
        logits = model(x) |> cpu # move to CPU # 1, l, b
        predictions = dropdims(argmax(logits, dims=1), dims=1) # l, b
        predictions_last_pixel = predictions[end, :]
        # convert CartesianIndex matrix to int matrix
        temp = zeros(Int, size(predictions_last_pixel))
        for idx in eachindex(predictions_last_pixel)
            temp[idx] = Int(predictions_last_pixel[idx][1] - 1)  # Extract and convert to Int
        end
        accuracy = sum(temp .== y[end, :]) / length(temp)
        push!(accuracy_list, accuracy)
        next!(test_progress; showvalues=[("Epoch ", epoch), ("Accuracy ", mean(accuracy_list))])
    end

    # Training loop
    loss_moving_avg = nothing
    α = 0.05

    train_progress = Progress(length(train_loader))
    for (x, y) in train_loader
        x, y = x |> device, y |> device

        function criterion(logits, y)
            onehot_y = Flux.onehotbatch(y, 0:n_classes-1)
            Flux.Losses.logitcrossentropy(logits, onehot_y)
        end

        loss, grads = Flux.withgradient(m -> criterion(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])

        # Update moving average of the loss
        if loss_moving_avg === nothing
            loss_moving_avg = loss
        else
            loss_moving_avg = α * loss + (1 - α) * loss_moving_avg
        end

        next!(train_progress; showvalues=[("Epoch ", epoch), ("Loss ", loss_moving_avg)])
    end

end
