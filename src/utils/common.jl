using Flux
using DelimitedFiles

count_params = (model) -> sum(length(p) for p in Flux.params(model))

function save_losses(experiment, train_losses, test_losses; path="saved_csv")
    mkpath("$path/$experiment")
    train_file = "$path/$experiment/train_losses.csv"
    test_file = "$path/$experiment/test_losses.csv"

    if isfile(train_file)
        rm(train_file)
        println("Deleted existing file: $train_file")
    end
    if isfile(test_file)
        rm(test_file)
        println("Deleted existing file: $test_file")
    end

    writedlm(train_file, train_losses, ',')
    writedlm(test_file, test_losses, ',')
    println("Saved new data to $train_file and $test_file")
end

function save_model_weights(model, experiment; path="saved_weights")
    mkpath("$path/$experiment")
    model_file = "$path/$experiment/model.bson"

    if isfile(model_file)
        rm(model_file)
        println("Deleted existing file: $model_file")
    end

    BSON.@save model_file model
    println("Saved model to $model_file")
end