include("./mnist.jl")
include("./shakespeare.jl")
include("./lra_retrieval.jl")
include("../models/mamba.jl")
include("../models/transformer.jl")

function get_dataloaders(dataset, device; p)
    if dataset == "mnist"
        train_loader, validation_loader = get_mnist_dataloaders(device;
            input_dim = p["input_dim"], train_batch_size=p["train_batch_size"], validation_batch_size=p["validation_batch_size"])
        vocab = nothing

    elseif dataset == "shakespeare"
        vocab, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len = p["seq_len"], data_to_use_percent = p["data_to_use_percent"])
		train_loader = Flux.DataLoader((trainX, trainY), batchsize = p["train_batch_size"], shuffle = true, partial = false) |> device
		validation_loader = Flux.DataLoader((testX, testY), batchsize = p["validation_batch_size"], shuffle = false, partial = false) |> device

    elseif dataset == "lra_retrieval"
        vocab, train_text_1, train_text_2, trainY, validation_text_1, validation_text_2, validationY =
            get_lra_retrieval(data_to_use_percent=params["data_to_use_percent"], seq_len=params["seq_len"])

        train_loader = Flux.DataLoader((train_text_1, train_text_2, trainY), batchsize=params["train_batch_size"], shuffle=true, partial=false) |> device
        validation_loader = Flux.DataLoader((validation_text_1, validation_text_2, validationY), batchsize=params["validation_batch_size"], shuffle=false, partial=false) |> device
    else
        error("$(p["dataset"]) is not a valid dataset name")
    end
    return train_loader, validation_loader, vocab
end

function get_criterion(dataset)
    if dataset == "mnist"
        criterion = (logits, y) -> begin
            onehot_y = Flux.onehotbatch(y, 0:9)
            Flux.Losses.logitcrossentropy(logits[:,end,:], onehot_y)
        end
    elseif dataset == "shakespeare"
        criterion = Flux.Losses.logitcrossentropy
    elseif dataset == "lra_retrieval"
        criterion = Flux.Losses.logitbinarycrossentropy
    else
        error("$(p["dataset"]) is not a valid dataset name")
    end
    return criterion
end

function get_model(p; vocab = nothing)
	
	if !isnothing(vocab)
		vocab_size = length(vocab)
	end

	if p["dataset"]=="lra_retrieval" && p["model_name"]=="mamba"
		model = MambaDualEncoder(vocab_size; embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"], 
			ssm_dropout=p["ssm_dropout"], use_A_dropout=p["use_A_dropout"], use_cuda_scan=p["use_cuda_scan"] )
	elseif p["dataset"]=="shakespeare" && p["model_name"] == "transformer"
        model = TransformerGPT(vocab, p["seq_len"], n_embed = p["embed_dim"], n_layers = p["n_layers"])
    elseif p["dataset"]=="shakespeare" && p["model_name"] == "mamba"
        model = MambaTextGenerator(vocab_size, embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"],
            ssm_dropout=p["ssm_dropout"], use_A_dropout=p["use_A_dropout"], use_cuda_scan=p["use_cuda_scan"])
	elseif p["dataset"]=="mnist" && p["model_name"] == "mamba"
		model = MambaClassifier(p["input_dim"], 10; embed_dim = p["embed_dim"],  N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"],
			ssm_dropout=p["ssm_dropout"], use_A_dropout=p["use_A_dropout"], use_cuda_scan=p["use_cuda_scan"] )
    else
        error("model $(p["model_name"]) is not implemented for dataset $(p["dataset"])")
    end
	return model
end

function get_accuracy(logits, y, dataset)
    if dataset == "mnist"
        predictions = dropdims(argmax(logits, dims=1), dims=1) # l, b
        predictions_last_element = predictions[end, :]
        # convert CartesianIndex matrix to int matrix
        temp = zeros(Int, size(predictions_last_element))
        for idx in eachindex(predictions_last_element)
            temp[idx] = Int(predictions_last_element[idx][1] - 1)  # Extract and convert to Int
        end
        accuracy = sum(temp .== y) / length(temp)
    elseif dataset == "shakespeare"
        #TODO
        accuracy = 0.0
    elseif dataset == "lra_retrieval"
        predictions = sigmoid.(logits[:, end, :]) .>= 0.5
        accuracy = mean(predictions .== y[:, end, :])
    else
        error("$(p["dataset"]) is not a valid dataset name")
    end
    return accuracy

end