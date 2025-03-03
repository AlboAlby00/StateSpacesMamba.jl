using DataFrames, Tables, Flux, DelimitedFiles, Base.Threads, BenchmarkTools, ProgressMeter
using Downloads, GZip, Serialization

function get_vocab(datasets; min_freq = 10, tokenizer="char")

    @assert tokenizer=="char" || tokenizer=="subword"
    
    T = tokenizer=="char" ? Char : String
	thread_counts = [Dict{T, Int}() for _ in 1:nthreads()]

	@threads for i in eachindex(datasets)
		tid = threadid()
		local_count = thread_counts[tid]

        tokens = tokenizer=="char" ? datasets[i] : split(datasets[i])

		for t in tokens
			local_count[t] = get(local_count, t, 0) + 1
		end
	end

	global_count = Dict{T, Int}()
	for local_count in thread_counts
		for (t, freq) in local_count
			global_count[t] = get(global_count, t, 0) + freq
		end
	end

	# Filter based on min freq
	vocab = [t for (t, freq) in global_count if freq >= min_freq]

    println("length of unfiltered vocab: $(length(vocab))")
    println("length of filtered vocab: $(length(global_count))")

	return sort(vocab)
end

function truncate_strings(strs, max_length = 4000)::Matrix{Char}
    truncated = [length(s) > max_length ? SubString(s, 1, max_length) : s for s in strs]
    max_len = maximum(length.(truncated))
    char_matrix = fill(' ', length(truncated), max_len)
    for i in eachindex(truncated)
        char_matrix[i, 1:length(truncated[i])] .= collect(truncated[i])
    end
    return char_matrix
end

function pad(charVector::Matrix{Char}, length::Int)
    current_length = size(charVector, 2)
    if current_length < length
        padding = fill(' ', 1, length - current_length)
        return hcat(charVector, padding)
    else
        return charVector
    end
end

function download_and_extract_dataset(; download_folder = "download", dataset_folder = "lra_release", url = "https://storage.googleapis.com/long-range-arena/lra_release.gz")
	gz_file_path = joinpath(download_folder, "$dataset_folder.gz")
	file_path = joinpath(download_folder, "$dataset_folder")
	println("Start downloading...")
	progress = nothing
	function download_progress_callback(total, now)
		# not working now
		if isnothing(progress)
			progress = Progress(total; desc = "Downloading...", dt = 1.0, output = stderr)
			println("creating bar")
		else
			ProgressMeter.update!(progress, now)
		end
	end

	#Downloads.download(url, gz_file_path, progress = download_progress_callback)
	println("Download completed...")
	println("Start extraction...")
	GZip.open(gz_file_path) do f_in
		open(file_path, "w") do f_out
			write(f_out, read(f_in))
		end
	end
	println("Extraction completed.")
	rm(gz_file_path)
end


function get_lra_retrival(; seq_len = 4000, data_to_use_percent = 1, vocab_folder = "vocab")

	if !isdir("download/lra_release")
		println("File not found.")
		download_and_extract_dataset()
	end

	println("Start loading dataset.")

    num_rows = count(_ -> true, eachline("download/lra_release/lra_release/tsv_data/new_aan_pairs.train.tsv"))
    train_text_1 = Matrix{Char}(undef, num_rows, 4000)
    train_text_2 = Matrix{Char}(undef, num_rows, 4000)
    trainY = Vector{Float32}(undef, num_rows)

    open("download/lra_release/lra_release/tsv_data/new_aan_pairs.train.tsv") do file

        for (i, line) in enumerate(eachline(file))
            
            parts = split(line, '\t')
            trainY[i] = parse(Float32, parts[1])
    
            truncated_1 = truncate_strings([parts[4]])  # Character matrix (1 row)
            truncated_2 = truncate_strings([parts[5]])  
            
            padded_1 = pad(truncated_1, seq_len)
            padded_2 = pad(truncated_2, seq_len)

            train_text_1[i,:] = padded_1
            train_text_2[i,:] = padded_2

        end
    end

    println("Training dataset loaded")

	validation_data = readdlm("download/lra_release/lra_release/tsv_data/new_aan_pairs.eval.tsv", '\t', header = false)

	validationY = validation_data[:, 1]
	validation_text_1 = truncate_strings(validation_data[:, 4])
	validation_text_2 = truncate_strings(validation_data[:, 5])

	println("Validation dataset loaded.")

	if !isfile("$vocab_folder/lra_retrival_vocab.jls")
		println("Vocab file not found, start computing vocab...")
		vocab = get_vocab(
			vcat(train_text_1, train_text_2, validation_text_1, validation_text_2);
			min_freq = 10)
		println("Vocab file computed.")
		open("$vocab_folder/lra_retrival_vocab.jls", "w") do io
			Serialization.serialize(io, vocab)
		end
	else
		println("Vocab file found")
		vocab = Serialization.deserialize("$vocab_folder/lra_retrival_vocab.jls")
	end

    validation_text_1 = map(c -> Int32(findfirst(==(c), vocab)), validation_text_1)
    validation_text_2 = map(c -> Int32(findfirst(==(c), vocab)), validation_text_2)
    train_text_1 = map(c -> Int32(findfirst(==(c), vocab)), train_text_1)
    train_text_2 = map(c -> Int32(findfirst(==(c), vocab)), train_text_2)

	return vocab, train_text_1, train_text_2, trainY, validation_text_1, validation_text_2, validationY
end

vocab, train_text_1, train_text_2, trainY, validation_text_1, validation_text_2, validationY = get_lra_retrival()


println("---")
