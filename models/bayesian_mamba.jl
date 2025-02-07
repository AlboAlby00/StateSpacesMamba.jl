using Random
using Flux: logitcrossentropy

include("./mamba.jl")

# utils functions

function softplus(x)
	return log.(1 .+ exp.(x))
end

function inverse_softplus(y)
	return log.(exp.(y) .- 1)
end

function log_gaussian_prob(x, mu, sigma)
	return -log.(sigma) .- (x .- mu) .^ 2 ./ (2 .* sigma .^ 2)
end

function gaussian_prob(x, mu, sigma)
	scaling = 1 ./ (sqrt.(2 * Float32(π) .* sigma .^ 2 ))
	bell = exp.(-(x .- mu).^2 / (2 .* sigma .^ 2 ) )
	return scaling .* bell
end

abstract type Prior end

# Gaussian Mixture Prior 

struct GaussianMixturePrior <: Prior
	alpha::Float32
	one_minus_alpha::Float32
	sigma_1::Float32
	sigma_2::Float32
end

function GaussianMixturePrior(alpha, sigma_1, sigma_2)
	one_minus_alpha = 1 - alpha
	return GaussianMixturePrior(alpha, one_minus_alpha, sigma_1, sigma_2)
end

function log_prob(prior::GaussianMixturePrior, model_params)
	log_probs = [
		sum(log.(prior.alpha .* gaussian_prob(model_param, 0, prior.sigma_1) .+ prior.one_minus_alpha .* gaussian_prob(model_param, 0, prior.sigma_2)))
		for model_param in model_params
	]
	total_log_prob = sum(log_probs)
end

# Variational Posterior

struct VariationalPosterior
	var_μ_array::AbstractArray
	var_ρ_array::AbstractArray # σ = softplus(ρ)
end

Flux.@layer VariationalPosterior trainable = (var_μ_array, var_ρ_array)

function VariationalPosterior(model::Chain, init_scale_μ = 0.05, init_scale_ρ = 0.01)
	var_ρ_init = softplus(init_scale_ρ)

	var_μ_array = []
	var_ρ_array = []

	for p in Flux.params(model)
		var_μ = init_scale_μ .* randn(Float32, size(p))
		var_ρ = fill(Float32(var_ρ_init), size(p))
		push!(var_μ_array, var_μ)
		push!(var_ρ_array, var_ρ)
	end
	return VariationalPosterior(var_μ_array, var_ρ_array)
end

function log_prob(posterior::VariationalPosterior, model_params)
	log_probs = [
		sum(log_gaussian_prob(model_param, var_μ, softplus(var_ρ)))
		for (var_μ, var_ρ, model_param)
		in zip(posterior.var_μ_array, posterior.var_ρ_array, model_params)]
	total_log_prob = sum(log_probs)
end

function sample_model_params(posterior::VariationalPosterior)

	model_params = Vector{Array{Float32}}(undef, length(posterior.var_μ_array))  
	for i in eachindex(posterior.var_μ_array)
		var_μ, var_ρ = posterior.var_μ_array[i], posterior.var_ρ_array[i]
		σ = softplus(var_ρ)
		model_params[i] = var_μ .+ σ .* randn(Float32, size(var_μ))  # Assign sampled values
	end
	return model_params
end

function bayes_by_backprop_loss(posterior::VariationalPosterior, prior, sampled_params, logits, y)
	neg_log_likelihood = logitcrossentropy(logits, y)
    prior_log_prob = log_prob(prior, sampled_params)
    var_posterior_log_prob = log_prob(posterior, sampled_params)
    kl_loss = var_posterior_log_prob .- prior_log_prob
    num_batches = size(y)[end]
    var_loss = neg_log_likelihood .+ kl_loss ./ num_batches
end

#= model = MambaGPT(128) |> f32
posterior = VariationalPosterior(model)
prior = GaussianMixturePrior(0.75, 0.001, 0.75)

sampled_params = sample_model_params(posterior)

x = rand(1:128, 16, 32)
y = model(x)
ŷ = Flux.onehotbatch(rand(1:128, 16, 32), 1:128)

bayes_by_backprop_loss(posterior, prior, sampled_params, ŷ, y)
 =#

