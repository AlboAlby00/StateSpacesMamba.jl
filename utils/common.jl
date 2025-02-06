using Flux
using DelimitedFiles

count_params = (model) -> sum(length(p) for p in Flux.params(model))

function save_losses(model_name, train_losses, test_losses; path="saved_csv")
    # Define file paths
    train_file = "$path/train_losses_$model_name.csv"
    test_file = "$path/test_losses_$model_name.csv"
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

function save_model_weights(model, model_name; path="saved_weights")
    model_file = "$path/$model_name.bson"
    if isfile(model_file)
        rm(model_file)
        println("Deleted existing file: $model_file")
    end
    BSON.@save model_file model
end