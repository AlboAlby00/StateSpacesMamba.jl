using Flux
using DelimitedFiles
using ArgParse

count_params = (model) -> sum(length(p) for p in Flux.params(model))
get_unix_time = () -> round(Int, time() * 1000)

function save_losses(experiment, train_losses, validation_losses, iteration; path="saved_csv")
    mkpath("$path/$experiment/$iteration")
    train_file = "$path/$experiment/$iteration/train_losses.csv"
    validation_file = "$path/$experiment/$iteration/validation_losses.csv"

    if isfile(train_file)
        rm(train_file)
        println("Deleted existing file: $train_file")
    end
    if isfile(validation_file)
        rm(validation_file)
        println("Deleted existing file: $validation_file")
    end

    writedlm(train_file, train_losses, ',')
    writedlm(validation_file, validation_losses, ',')
    println("Saved new data to $train_file and $validation_file")
end

function save_model_weights(model, experiment, iteration; path="saved_weights")
    mkpath("$path/$experiment")
    model_file = "$path/$experiment/model_$iteration.bson"

    if isfile(model_file)
        rm(model_file)
        println("Deleted existing file: $experiment/$model_file")
    end

    BSON.@save model_file model
    println("Saved model to $experiment/$model_file")
end

function set_seed(seed)
    Random.seed!(seed)
    CUDA.seed!(seed)
end

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--use_mlflow"
            help = "Enable MLflow logging"
            action = :store_true
        "experiment_list"
            help = "List of experiments (separated by spaces)"
            nargs = '+'
            arg_type = String
    end
    return parse_args(s)
end
