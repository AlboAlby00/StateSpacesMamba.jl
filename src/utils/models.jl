include("../models/mamba.jl")
include("../models/transformer.jl")

function get_model(p; vocab = nothing)
	
	if !isnothing(vocab)
		vocab_size = length(vocab)
	end

	if p["dataset"]=="lra_retrieval" && p["model_name"]=="mamba"
		model = MambaDualEncoder(vocab_size; embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"], 
			ssm_dropout=p["ssm_dropout"], use_A_dropout=p["use_A_dropout"], use_cuda_scan=p["use_cuda_scan"] )
	elseif p["dataset"]=="shakespeare" && params["model_name"] == "transformer"
        model = TransformerGPT(vocab, params["seq_len"], n_embed = p["embed_dim"], n_layers = p["n_layers"])
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
