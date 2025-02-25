using DataFrames, Tables, Flux, DelimitedFiles, Base.Threads, BenchmarkTools

function get_vocab_parallel(datasets; min_freq=10)
    thread_counts = [Dict{Char, Int}() for _ in 1:nthreads()]  # One dictionary per thread
    
    @threads for i in eachindex(datasets)
        tid = threadid()
        local_count = thread_counts[tid]
        
        for char in datasets[i]
            local_count[char] = get(local_count, char, 0) + 1
        end
    end

    global_count = Dict{Char, Int}()
    for local_count in thread_counts
        for (char, freq) in local_count
            global_count[char] = get(global_count, char, 0) + freq
        end
    end

    # Filter based on min_freq
    vocab = [char for (char, freq) in global_count if freq >= min_freq]

    return sort(vocab)
end

function get_vocab(datasets; min_freq=10)
    count = Dict{Char, Int}()
    for text in datasets
        for char in text
            count[char] = get(count, char, 0) + 1
        end
    end

    # Filter out characters with frequency less than min_freq
    vocab = [char for (char, freq) in count if freq >= min_freq]

    return sort(vocab)
end

function truncate_strings(strs, max_length=4000)
    return [length(s) > max_length ? SubString(s, 1, max_length) : s for s in strs]
end

function get_lra_retrival(; seq_len=4000, test_percent=0.2, data_to_use_percent=1)

    if !isfile("download/lra_release/lra_release/tsv_data/new_aan_pairs.eval.tsv")
        throw(ErrorException("File was not found"))
    end

    if !isfile("download/lra_release/lra_release/tsv_data/new_aan_pairs.train.tsv")
        throw(ErrorException("File was not found"))
    end

    train_text_1 = String[]
    train_text_2 = String[]
    testY = Float32[]
    open("download/lra_release/lra_release/tsv_data/new_aan_pairs.train.tsv") do file
        for line in eachline(file)
            parts = split(line, '\t')
            push!(testY, parse(Float32,parts[1]))
            push!(train_text_1, truncate_strings([parts[4]])[1])
            push!(train_text_2, truncate_strings([parts[5]])[1])
        end
    end

    validation_data = readdlm("download/lra_release/lra_release/tsv_data/new_aan_pairs.eval.tsv", '\t', header=false)

    validationY = validation_data[:,1]
    validation_text_1 = truncate_strings(validation_data[:,4])
    validation_text_2 = truncate_strings(validation_data[:,5])

    vocab = get_vocab_parallel(vcat(
        train_text_1,train_text_2, 
        validation_text_1, validation_text_2) )

    return vocab, train_text_1, train_text_2, validation_text_1, validation_text_2
end

get_lra_retrival()
println("---")