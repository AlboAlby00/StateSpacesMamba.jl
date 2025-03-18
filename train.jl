using Pkg
Pkg.activate(".")
Pkg.instantiate()

using Flux, Optimisers, ProgressMeter, MLFlowClient, Zygote
using Statistics, MLDatasets, CUDA, Revise, Plots, DelimitedFiles, BSON, YAML

CUDA.allowscalar(false)

include("src/utils/mnist.jl")
include("src/utils/common.jl")
include("src/utils/get_functions.jl")
include("src/utils/params.jl")

function train_and_evaluate(hp, train_loader, validation_loader, model, criterion; 
    mlflow_experiment_id=nothing, run_name="NO NAME")

    lr_decay_factor = (hp["init_fin_lr_ratio"])^(1 / hp["num_epochs"])

    if !isnothing(mlflow_experiment_id)
        exprun = createrun(MLF, mlflow_experiment_id; start_time=get_unix_time())
        # Log params
        for (name, value) in hp
            logparam(MLF, exprun, name, string(value))
        end
    end

    opt = Flux.Optimise.Optimiser(Flux.Optimise.ClipValue(1e-3), Flux.Optimise.Adam(hp["initial_lr"]))
    opt_state = Optimisers.setup(opt, model)

    best_model = deepcopy(model)
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
        for (x, y) in validation_loader
            y = y |> cpu
            logits = model(x) |> cpu
            validation_loss = criterion(logits, y)
            validation_accuracy = get_accuracy(logits, y, hp["dataset"])
            if isnan(validation_loss) || isnan(validation_accuracy)
                println("NaN value during validation")
                model = best_model
                continue
            end
            push!(losses, validation_loss)
            push!(accuracies, validation_accuracy)
            next!(validation_progress; showvalues=[("Mean Loss", mean(losses)), ("Accuracy", mean(accuracies))])
        end
        epoch_validation_loss = mean(losses)
        epoch_validation_accuracy = mean(accuracies)
        if !isnothing(mlflow_experiment_id)
            logmetric(MLF, exprun, "validation loss", Float64(epoch_validation_loss))
            logmetric(MLF, exprun, "validation accuracy", Float64(epoch_validation_accuracy))
        end
        push!(validation_losses, epoch_validation_loss)

        if epoch_validation_loss < best_validation_loss
            best_validation_loss = epoch_validation_loss
            best_model = deepcopy(model)
        end

        Flux.trainmode!(model)
        loss_moving_avg = nothing
        α = 0.05
        train_progress = Progress(length(train_loader), desc="Training Epoch $epoch")
        for (x, y) in train_loader

            loss, grads = Flux.withgradient(m -> criterion(m(x), y), model)
            
            if isnan(loss)
                println("NaN value during Training")
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
parsed_args = parse_commandline()

if parsed_args["use_mlflow"]
    MLF = MLFlow("http://localhost:8080/api")
end

# Run for multiple models
for experiment in parsed_args["experiment_list"]
#for experiment in ["mnist/mamba_ssm_dropout_evaluation"]
    experiment_yaml = YAML.load_file("experiments/$experiment.yaml")

    experiment_id = parsed_args["use_mlflow"] ? createexperiment(MLF, experiment) : nothing

    for (desc, params) in generate_combinations(experiment_yaml)

        println(desc)

        train_loader, validation_loader, vocab = get_dataloaders(params["dataset"], device; p=params)
        criterion = get_criterion(params["dataset"])

        for iteration in 1:params["num_iterations"]
            set_seed(iteration)
            println("start iteration $iteration")

            model = get_model(params; vocab=vocab) |> f32 |> device
            println("model for experiment '$experiment' has $(count_params(model)) parameters")

            run_name = "iteration=$iteration, $desc"

            train_losses, validation_losses, best_model = train_and_evaluate(
                params, train_loader, validation_loader, model, criterion; 
                mlflow_experiment_id=experiment_id, run_name=run_name)

            if params["save_csv"]
                save_losses(experiment, train_losses, validation_losses, iteration)
            end

            if params["save_model"]
                save_model_weights(best_model, experiment, iteration)
            end
        end

    end

end