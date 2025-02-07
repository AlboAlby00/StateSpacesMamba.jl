using Flux, Optimisers, ProgressMeter
using Statistics
using MLDatasets
using CUDA
using Revise
using Plots
using DelimitedFiles
using BSON

CUDA.allowscalar(false)

include("utils/shakespeare.jl")
include("utils/common.jl")
include("utils/models_dict.jl")

device = gpu_device()

train_batch_size = 64
test_batch_size = 64
model_name = "transformer_with_dropout"
seq_len = 256
save_csv = true
save_model = true

best_model = nothing
best_test_loss = Inf

alphabet, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len=seq_len)
vocab_size = length(alphabet)

train_loader = Flux.DataLoader((trainX, trainY), batchsize=train_batch_size, shuffle=true, partial=false) |> device
test_loader = Flux.DataLoader((testX, testY), batchsize=test_batch_size, shuffle=false, partial=false) |> device

model = models[model_name] |> f32 |> device

for (name, model) in models
    println("Number of parameters of model $name is $(count_params(model))")
end

# Learning rate scheduling
initial_lr = 1e-3
final_lr = 1e-4
num_epochs = 100
lr_decay_factor = (final_lr / initial_lr)^(1 / num_epochs)

opt = Optimisers.Adam(initial_lr)
opt_state = Optimisers.setup(opt, model)

function criterion(logits, y)
    Flux.Losses.logitcrossentropy(logits, y)
end

test_losses = []
train_losses = []

@showprogress for epoch in 1:num_epochs
    global best_test_loss
    global best_model

    # Adjust learning rate
    new_lr = initial_lr * lr_decay_factor^(epoch - 1)
    Optimisers.adjust!(opt_state, new_lr)

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
    epoch_test_loss = mean(losses)
    push!(test_losses, epoch_test_loss)

    if epoch_test_loss < best_test_loss
        best_test_loss = epoch_test_loss
        best_model = deepcopy(model)  # Save a copy of the best model
    end

    println(generate(model, alphabet, "The ", 200, seq_len))

    # Training loop
    Flux.trainmode!(model)
    loss_moving_avg = nothing
    α = 0.05
    train_progress = Progress(length(train_loader))
    for (x, y) in train_loader
        loss, grads = Flux.withgradient(m -> criterion(m(x), y), model)
        Flux.update!(opt_state, model, grads[1])
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
    save_losses(model_name, train_losses, test_losses)
end

if save_model
    save_model_weights(best_model, model_name)
end