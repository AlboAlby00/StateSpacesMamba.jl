include("../models/mamba.jl")
include("../models/transformer.jl")

function get_model(p; vocab = nothing)
	dataset = get(p, "dataset", nothing)
	model_name = get(p, "model_name", nothing)

	@assert !isnothing(vocab) || dataset == "mnist" "vocab is required for dataset $dataset"

	if dataset == "lra_retrieval"
		if model_name == "mamba"
			return MambaDualEncoder(length(vocab);
				embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"],
				ssm_dropout = p["ssm_dropout"], use_A_dropout = p["use_A_dropout"], use_cuda_scan = p["use_cuda_scan"])
		elseif model_name == "transformer"
			return TransformerDualEncoder(vocab, p["seq_len"], n_embed = p["embed_dim"], n_layers = p["n_layers"], 
				qk_dim = p["qk_dim"], v_dim = p["v_dim"], n_heads = p["n_heads"], dropout = p["dropout"])
		end
	elseif dataset == "shakespeare"
		if model_name == "transformer"
			return TransformerGPT(vocab, p["seq_len"], n_embed = p["embed_dim"], n_layers = p["n_layers"], 
				qk_dim = p["qk_dim"], v_dim = p["v_dim"], n_heads = p["n_heads"], dropout = p["dropout"])
		elseif model_name == "mamba"
			return MambaTextGenerator(length(vocab);
				embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"],
				ssm_dropout = p["ssm_dropout"], use_A_dropout = p["use_A_dropout"], use_cuda_scan = p["use_cuda_scan"])
		end
	elseif dataset == "mnist" && model_name == "mamba"
		return MambaClassifier(p["input_dim"], 10;
			embed_dim = p["embed_dim"], N = p["N"], n_layers = p["n_layers"], dropout = p["dropout"],
			ssm_dropout = p["ssm_dropout"], use_A_dropout = p["use_A_dropout"], use_cuda_scan = p["use_cuda_scan"])
	end

	error("Model $model_name is not implemented for dataset $dataset")
end
