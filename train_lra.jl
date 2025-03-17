using Flux, Optimisers, ProgressMeter, MLFlowClient
using Statistics, MLDatasets, CUDA, Revise, Plots, DelimitedFiles, BSON, YAML

CUDA.allowscalar(false)

include("src/utils/lra_retrieval.jl")
include("src/utils/common.jl")
include("src/utils/models.jl")
include("src/utils/params.jl")


function train_and_evaluate(hp, train_loader, validation_loader, model; mlflow_experiment_id=nothing, run_name="NO NAME")

    lr_decay_factor = (hp["init_fin_lr_ratio"])^(1 / hp["num_epochs"])

    if !isnothing(mlflow_experiment_id)
        exprun = createrun(MLF, mlflow_experiment_id; start_time=get_unix_time())
        # Log params
        for (name, value) in hp
            logparam(MLF, exprun, name, string(value))
        end
    end

    opt = Optimisers.Adam(hp["initial_lr"])
    opt_state = Optimisers.setup(opt, model)

    best_model = nothing
    best_validation_loss = Inf

    validation_losses = []
    train_losses = []

    for epoch in 1:hp["num_epochs"]

        new_lr = hp["initial_lr"] * lr_decay_factor^(epoch - 1)
        Optimisers.adjust!(opt_state, new_lr)

        Flux.testmode!(model)

        validation_progress = Progress(length(validation_loader), desc="Validating Epoch $epoch")
        losses = []
        accuracies = []

        for (x1, x2, y) in validation_loader
            l, _ = size(x1)
            #y = repeat(reshape(y, 1, 1, :), 1, l, 1)  # Reshape and repeat so that size(y) == size(logits)
            y = y |> cpu
            logits = model(x1, x2) |> cpu
            validation_loss = Flux.Losses.logitbinarycrossentropy(logits, y)

            predictions = sigmoid.(logits[:, end, :]) .>= 0.5
            acc = mean(predictions .== y[:, end, :])
            push!(accuracies, acc)
            push!(losses, validation_loss)
            next!(validation_progress; showvalues=[("Mean Loss", mean(losses)), ("Accuracy", mean(accuracies))])

        end
        epoch_validation_loss = mean(losses)
        epoch_validation_accuracy = mean(accuracies)
        push!(validation_losses, epoch_validation_loss)
        if !isnothing(mlflow_experiment_id)
            logmetric(MLF, exprun, "validation loss", Float64(epoch_validation_loss))
            logmetric(MLF, exprun, "validation accuracy", Float64(epoch_validation_accuracy))
        end

        if epoch_validation_loss < best_validation_loss
            best_validation_loss = epoch_validation_loss
            best_model = deepcopy(model)
        end

        Flux.trainmode!(model)
        loss_moving_avg = nothing
        α = 0.5
        train_progress = Progress(length(train_loader), desc="Training Epoch $epoch")
        for (x1, x2, y) in train_loader
            l, _ = size(x1)

            loss, grads = Flux.withgradient(m -> Flux.Losses.logitbinarycrossentropy(m(x1, x2), y), model)
            if loss == NaN
                println("NaN value")
                model = best_model
                continue
            end
            Flux.update!(opt_state, model, grads[1])
            if loss_moving_avg === nothing
                loss_moving_avg = loss
            else
                loss_moving_avg = α * loss + (1 - α) * loss_moving_avg
            end
            push!(train_losses, loss)
            next!(train_progress; showvalues=[("Loss", loss_moving_avg)])
            if !isnothing(mlflow_experiment_id)
                logmetric(MLF, exprun, "train loss", Float64(loss))
            end
        end
    end

    if !isnothing(mlflow_experiment_id)
        updaterun(MLF, exprun; status=MLFlowClient.FINISHED, end_time=get_unix_time(), run_name=run_name)
    end

    return train_losses, validation_losses, best_model
end


device = gpu_device()
use_mlflow = true

if use_mlflow
    MLF = MLFlow("http://localhost:8080/api")
end

# Run for multiple models
for experiment in ["lra_retrieval/mamba_test_ssm_dropout_small_dataset"]
    experiment_yaml = YAML.load_file("experiments/$experiment.yaml")

    experiment_id = use_mlflow ? createexperiment(MLF, experiment) : nothing

    for (desc, params) in generate_combinations(experiment_yaml)

        println(desc)

        vocab, train_text_1, train_text_2, trainY, validation_text_1, validation_text_2, validationY =
            get_lra_retrieval(data_to_use_percent=params["data_to_use_percent"], seq_len=params["seq_len"])

        train_loader = Flux.DataLoader((train_text_1, train_text_2, trainY), batchsize=params["train_batch_size"], shuffle=true, partial=false) |> device
        validation_loader = Flux.DataLoader((validation_text_1, validation_text_2, validationY), batchsize=params["test_batch_size"], shuffle=false, partial=false) |> device

        for iteration in 1:params["num_iterations"]
            set_seed(iteration)
            println("start iteration $iteration")

            model = get_model(params; vocab=vocab) |> f32 |> device
            println("model for experiment '$experiment' has $(count_params(model)) parameters")

            run_name = "iteration=$iteration, $desc"
            train_losses, validation_losses, best_model = train_and_evaluate(params, train_loader, validation_loader, model; mlflow_experiment_id=experiment_id, run_name=run_name)

            if params["save_csv"]
                save_losses(experiment, train_losses, validation_losses, iteration)
            end

            if params["save_model"]
                save_model_weights(best_model, experiment, iteration)
            end
        end

    end

end
