using Flux, Optimisers, ProgressMeter
using Statistics, MLDatasets, CUDA, Revise, Plots, DelimitedFiles, BSON

CUDA.allowscalar(false)

include("utils/shakespeare.jl")
include("utils/common.jl")
include("utils/models_dict.jl")
include("models/bayesian_mamba.jl")

device = gpu_device()

function train_and_evaluate(model_name, train_batch_size=64, test_batch_size=64, seq_len=256, num_epochs=100, initial_lr=1e-3, final_lr=1e-4, save_csv=true, save_model=true)
    best_model = nothing
    best_test_loss = Inf

    alphabet, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len=seq_len)
    vocab_size = length(alphabet)

    train_loader = Flux.DataLoader((trainX, trainY), batchsize=train_batch_size, shuffle=true, partial=false) |> device
    test_loader = Flux.DataLoader((testX, testY), batchsize=test_batch_size, shuffle=false, partial=false) |> device

    model = models[model_name] |> f32 |> device

    println("Number of parameters of model $model_name is $(count_params(model))")

    # Learning rate scheduling
    lr_decay_factor = (final_lr / initial_lr)^(1 / num_epochs)
    opt = Optimisers.Adam(initial_lr)
    opt_state = Optimisers.setup(opt, model)

    function criterion(logits, y)
        Flux.Losses.logitcrossentropy(logits, y)
    end

    test_losses = []
    train_losses = []
    loss_moving_avg = 6.0  # Initialize loss moving average

    for epoch in 1:num_epochs
        global best_test_loss, best_model, loss_moving_avg

        # Adjust learning rate
        new_lr = initial_lr * lr_decay_factor^(epoch - 1)
        Optimisers.adjust!(opt_state, new_lr)

        # Validation loop
        Flux.testmode!(model)
        test_progress = Progress(length(test_loader), desc="Validating Epoch $epoch")
        losses = []
        for (x, y) in test_loader
            y = y |> cpu
            logits = model(x) |> cpu
            test_loss = criterion(logits, y)
            push!(losses, test_loss)
            next!(test_progress; showvalues=[("Mean Loss", mean(losses))])
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
        
        α = 0.05
        train_progress = Progress(length(train_loader), desc="Training Epoch $epoch")
        for (x, y) in train_loader
            loss, grads = Flux.withgradient(m -> criterion(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
            loss_moving_avg = α * loss + (1 - α) * loss_moving_avg
            push!(train_losses, loss)
            next!(train_progress; showvalues=[("Loss", loss_moving_avg)])
        end
    end

    if save_csv
        save_losses(model_name, train_losses, test_losses)
    end

    if save_model
        save_model_weights(best_model, model_name)
    end
end

# Run for multiple models
for model_name in ["mamba", "transformer"]
    train_and_evaluate(model_name)
end