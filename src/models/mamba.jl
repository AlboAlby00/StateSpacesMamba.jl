using Flux

include("../scan/utils.jl")
include("../scan/scan.jl")
include("./ssm.jl")
include("classifier_head_dual.jl")

function FullyConnectedBlock(embed_dim, dropout = 0.1)
	return Chain(
		Dense(embed_dim => embed_dim, swish),
		Dropout(dropout),
		LayerNorm(embed_dim),
	)
end

function MambaEncoder(embed_dim, n_layers, n_fc_layers, N, expand, kernel_size, dropout, ssm_dropout, use_A_dropout, use_cuda_scan; Δrank = nothing)
	if isnothing(Δrank)
		Δrank = embed_dim ÷ 8
	end
	return Chain(
		[
			SkipConnection(MambaBlock(embed_dim, embed_dim, D = embed_dim * expand, N = N, Δrank = Δrank, dropout = dropout, kernel_size = kernel_size,
				ssm_dropout = ssm_dropout, use_A_dropout = use_A_dropout, use_cuda_scan = use_cuda_scan), +) for _ in 1:n_layers
		]...,
		[SkipConnection(FullyConnectedBlock(embed_dim, dropout), +) for _ in 1:n_fc_layers]...,
	)
end

function MambaClassifier(in, out; embed_dim = 64, n_layers = 3, N = 16, kernel_size = 5, expand = 2,
	dropout = 0.0, ssm_dropout = 0.0, use_A_dropout = true, use_cuda_scan = true)
	model = Chain(
		MambaBlock(in, embed_dim, D = embed_dim * expand, N = N, kernel_size = kernel_size, dropout = dropout, ssm_dropout = ssm_dropout, Δrank = embed_dim ÷ 8,
            use_A_dropout = use_A_dropout, use_cuda_scan = use_cuda_scan, use_causal_conv = false, use_normalization=false),
		[MambaBlock(embed_dim, embed_dim, D = embed_dim * expand, N = N, Δrank = embed_dim ÷ 8, dropout = dropout, kernel_size = kernel_size, 
            ssm_dropout = ssm_dropout, use_A_dropout = use_A_dropout, use_cuda_scan = use_cuda_scan, use_normalization=false, use_causal_conv = false) for _ in 1:n_layers-1]...,
		Dense(embed_dim => out),
	)
	return model
end

function MambaTextGenerator(vocab_size; embed_dim = 128, n_layers = 3, n_fc_layers = 2, N = 16, expand = 2, kernel_size = 4,
	dropout = 0.0, ssm_dropout = 0.0, use_A_dropout = true, use_cuda_scan = true)
	model = Chain(
		Embedding(vocab_size, embed_dim),
		Dropout(dropout),
		MambaEncoder(embed_dim, n_layers, n_fc_layers, N, expand, kernel_size, dropout, ssm_dropout, use_A_dropout, use_cuda_scan),
		Dense(embed_dim => vocab_size),
	)
	return model
end

# Define the Mamba dual encoder for the LRA Document Retrieval task
struct MambaDualEncoder
	embedding::Embedding
	shared_encoder::Chain
	classifier_head::ClassifierHeadDual
	dropout::Dropout
end
Flux.@layer MambaDualEncoder trainable = (embedding, shared_encoder, classifier_head)

function MambaDualEncoder(vocab_size; embed_dim = 128, n_layers = 3, n_fc_layers = 2, N = 16, expand = 2, kernel_size = 5,
	dropout = 0.0, ssm_dropout = 0.0, use_A_dropout = true, use_cuda_scan = true)

	embedding = Embedding(vocab_size, embed_dim)
	shared_encoder = MambaEncoder(embed_dim, n_layers, n_fc_layers, N, expand, kernel_size, dropout, ssm_dropout, use_A_dropout, use_cuda_scan)
	classifier_head = ClassifierHeadDual(embed_dim, embed_dim, 1)
	dropout_layer = Dropout(dropout)

	return MambaDualEncoder(embedding, shared_encoder, classifier_head, dropout_layer)
end

function (m::MambaDualEncoder)(x1, x2)
	out1 = m.shared_encoder(m.dropout(m.embedding(x1)))
	out2 = m.shared_encoder(m.dropout(m.embedding(x2)))
	out = m.classifier_head(out1, out2)
	return out
end

struct MambaBlock
	norm::LayerNorm
	project_input::Dense
	project_res::Dense
	conv_1d::Conv
	ssm::SSM
	project_output::Dense
	dropout::Dropout
	use_normalization::Bool
end

# Constructor for MambaBlock
function MambaBlock(input_dim, output_dim; D = 64, N = 16, kernel_size = 4, dropout = 0.0, Δrank = 32, ssm_dropout = 0.0, use_A_dropout = true, use_cuda_scan = true, use_normalization = true, use_causal_conv = true)
	norm = LayerNorm(input_dim, relu)
	project_input = Dense(input_dim => D)
	project_res = Dense(input_dim => D)
	if use_causal_conv
		# pad defined so that input D == output D and applied only on the left to ensure no peeking in the future
		conv_1d = Conv((kernel_size,), D => D, relu, pad = (kernel_size - 1, 0))
	else
		@assert kernel_size % 2 == 1 "Kernel size must be odd if not using causal convolution"
		conv_1d = Conv((kernel_size,), D => D, relu, pad = (kernel_size ÷ 2,)) # pad defined so that input D == output D, kernel size must be odd
	end
	ssm = SSM(D = D, N = N, Δrank = Δrank, ssm_dropout = ssm_dropout, use_A_dropout = use_A_dropout, use_cuda_scan = use_cuda_scan)
	project_output = Dense(D => output_dim)
	dropout = Dropout(dropout)
	return MambaBlock(norm, project_input, project_res, conv_1d, ssm, project_output, dropout, use_normalization)
end

# Forward pass for MambaBlock
function (m::MambaBlock)(x)

	d, l, b = size(x)
	if m.use_normalization
		x = m.norm(x)
	end

	out_project = m.project_input(x)
	# Conv layer requires size l, d, b but out_project size is d, l, b. We need to permute the dims
	out_project = permutedims(out_project, (2, 1, 3)) # permute l and d
	out_conv = swish(m.conv_1d(out_project))
	out_conv = permutedims(out_conv, (2, 1, 3)) # permute d and l to return to original size

	out_ssm = swish(m.ssm(out_conv))

	# inner residual connection
	out_res = out_ssm .+ swish(m.project_res(x))

	out = m.project_output(out_res)

	return m.dropout(out)
end

Flux.@layer MambaBlock trainable = (project_input, project_res, conv_1d, ssm, project_output)
