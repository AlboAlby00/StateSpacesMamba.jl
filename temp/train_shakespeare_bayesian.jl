using Flux, Optimisers, ProgressMeter
using Statistics
using MLDatasets
using CUDA
using Revise
using Plots
using DelimitedFiles
using BSON
import ChainRulesCore: @ignore_derivatives

CUDA.allowscalar(false)

include("utils/shakespeare.jl")
include("utils/common.jl")
include("models/bayesian_mamba.jl")
include("models/mamba.jl")
include("models/transformer.jl")

device = cpu_device()

train_batch_size = 64
test_batch_size = 64
model_name = "bayesian_mamba"
seq_len = 32
save_csv = true
save_model = true

best_model = nothing
best_test_loss = Inf

alphabet, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len=seq_len, data_to_use_percent=0.05)
vocab_size = length(alphabet)

function set_params_manually!(model, new_params)
    params = Flux.params(model)
    for (p, new_p) in zip(params, new_params)
        @assert size(p) == size(new_p) "Parameter count mismatch"
        if(ndims(p)<3)
            p .= new_p
        end
    end
end

models = Dict(
    "transformer" => TransformerGPT(alphabet, seq_len, n_embed=512, n_layers=6), # 1987139 parameters
    "transformer_with_dropout" => TransformerGPT(alphabet, seq_len, n_embed=512, n_layers=6, dropout=0.1), 
    "mamba" => MambaGPT(vocab_size, embed_dim = 128, N = 16, n_layers=6), # 1927859 parameters
    "mamba_with_dropout" => MambaGPT(vocab_size, embed_dim = 128, N = 16, n_layers=6, dropout=0.1),
    "transformer_small" => TransformerGPT(alphabet, seq_len, n_embed=256, n_layers=3),
    "mamba_small" => MambaGPT(vocab_size, embed_dim = 128, N = 8, n_layers=3),
    "mamba_small_with_dropout" => MambaGPT(vocab_size, embed_dim = 128, N = 8, n_layers=3, dropout=0.1),
    "bayesian_mamba" => MambaGPT(vocab_size, embed_dim = 8, N = 4, n_layers=3)
)

train_loader = Flux.DataLoader((trainX, trainY), batchsize=train_batch_size, shuffle=true, partial=false) |> device
test_loader = Flux.DataLoader((testX, testY), batchsize=test_batch_size, shuffle=false, partial=false) |> device

model = models[model_name] |> f32 |> device
prior = GaussianMixturePrior(0.75, 0.0001, 0.1)
posterior = VariationalPosterior(model)

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

test_losses = []
train_losses = []

loss_moving_avg = 5.0

@showprogress for epoch in 1:num_epochs
    global best_test_loss
    global best_model
    global loss_moving_avg

    # Adjust learning rate
    new_lr = initial_lr * lr_decay_factor^(epoch - 1)
    Optimisers.adjust!(opt_state, new_lr)

    # Validation loop
    Flux.testmode!(model)
    test_progress = Progress(length(test_loader))
    losses = []
    for (x, y) in test_loader
        y = y |> cpu
        sampled_params = sample_model_params(posterior)
        logits = model(x) |> cpu
        #println(model)
        set_params_manually!(model, sampled_params)
        #println(model)
        test_loss = bayes_by_backprop_loss(posterior,prior,sampled_params,logits,y)
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
    loss_moving_avg = 5.0
    α = 0.05
    train_progress = Progress(length(train_loader))
    for (x, y) in train_loader
        sampled_params = sample_model_params(posterior)
        @assert all(layer -> all(isfinite, layer), sampled_params) "sampled_params contains Inf or NaN values!"
        set_params_manually!(model, sampled_params)
        loss, grads = Flux.withgradient(
            var -> bayes_by_backprop_loss(var, prior, sampled_params, model(x), y), posterior)
    
    
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