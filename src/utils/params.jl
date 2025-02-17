using IterTools
using YAML

function parse_config(filename)
    params = YAML.load_file(filename)
    return Dict{String, Any}(params)
end

function generate_combinations(params)
    keys = String[]
    values = []
    fixed_params = Dict{String, Any}()
    
    for (k, v) in params
        if isa(v, AbstractArray)
            push!(keys, k)
            push!(values, v)
        else
            fixed_params[k] = v
        end
    end
    
    combinations = [merge(fixed_params, Dict(k => v for (k, v) in zip(keys, comb))) for comb in IterTools.product(values...)]
    
    descriptions = [join(["$k=$v" for (k, v) in comb if k in keys], ", ") for comb in combinations]
    
    return [(desc, comb) for (comb, desc) in zip(combinations, descriptions)]
end

#= # Example usage
config_file = "experiments/test.yaml"
params = parse_config(config_file)
combinations_with_descriptions = generate_combinations(params)
println(combinations_with_descriptions[1]) =#



