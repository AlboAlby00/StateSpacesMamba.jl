using Flux

include("../scan/utils.jl")
include("../scan/scan.jl")
include("./ssm.jl")

function Mamba(in, out; mamba_block_output_dim = 64, n_layers = 3, N = 16, expand = 16, kernel_size = 4, dropout = 0.0)
	model = Chain(
		MambaBlock(in, mamba_block_output_dim, D = expand * in, N = N, dropout = dropout),
		[SkipConnection(MambaBlock(mamba_block_output_dim, mamba_block_output_dim, D = expand * in, N = N, Δrank = in, dropout = dropout, kernel_size=kernel_size), +) for _ in 1:n_layers-1]...,
		Dense(mamba_block_output_dim => out),
	)
	return model
end

function MambaGPT(vocab_size; embed_dim = 128, n_layers = 3, N = 16, expand = 2, kernel_size = 4, dropout = 0.0)
	model = Chain(
		Embedding(vocab_size, embed_dim),
		[SkipConnection(MambaBlock(embed_dim, embed_dim, D = embed_dim * expand, N = N, Δrank = embed_dim ÷ 8, dropout = dropout, kernel_size = kernel_size), +) for _ in 1:n_layers-1]...,
		Dense(embed_dim => vocab_size),
	)
	return model
end

struct MambaBlock
	norm::LayerNorm
	project_input::Dense
	project_res::Dense
	causal_conv_1d::Conv
	ssm::SSM
	project_output::Dense
	dropout::Dropout
end

# Constructor for MambaBlock
function MambaBlock(input_dim, output_dim; D = 64, N = 16, kernel_size = 4, dropout = 0.0, Δrank = 32)
	norm = LayerNorm(input_dim, relu)
	project_input = Dense(input_dim => D)
	project_res = Dense(input_dim => D)
	# pad defined so that input D == output D and applied only on the left to ensure no peeking in the future
	causal_conv_1d = Conv((kernel_size,), D => D, relu, pad = (kernel_size - 1, 0))
	ssm = SSM(D = D, N = N, Δrank = Δrank)
	project_output = Dense(D => output_dim)
	dropout = Dropout(dropout)
	return MambaBlock(norm, project_input, project_res, causal_conv_1d, ssm, project_output, dropout)
end

# Forward pass for MambaBlock
function (m::MambaBlock)(x)

	d, l, b = size(x)

	x = m.norm(x)

	out_project = m.project_input(x)
	# Conv layer requires size l, d, b but out_project size is d, l, b. We need to permute the dims
	out_project = permutedims(out_project, (2, 1, 3)) # permute l and d
	#out_conv = swish(out_project)
	out_conv = swish(m.causal_conv_1d(out_project))
	out_conv = permutedims(out_conv, (2, 1, 3)) # permute d and l to return to original size

	out_ssm = swish(m.ssm(out_conv))

	# inner residual connection
	out_res = out_ssm .+ swish(m.project_res(x))

	out_drop = m.dropout(out_res)

	out = m.project_output(out_drop)

	return out
end

Flux.@layer MambaBlock trainable = (project_input, project_res, causal_conv_1d, ssm, project_output)
