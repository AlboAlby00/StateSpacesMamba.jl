using Flux

count_params = (model) -> sum(length(p) for p in Flux.params(model))