using DataFrames, Tables, Flux, DelimitedFiles, Base.Threads, BenchmarkTools, ProgressMeter
using Downloads, GZip, Serialization, Tar, CodecZlib

function get_vocab(datasets; min_freq = 10)

	T = typeof(datasets[1][1, 1])
	thread_counts = [Dict{T, Int}() for _ in 1:nthreads()]

	@threads for i in eachindex(datasets)
		tid = threadid()
		local_count = thread_counts[tid]

		tokens = datasets[i]

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

	println("length of filtered vocab: $(length(vocab))")
	println("length of unfiltered vocab: $(length(global_count))")

	return sort(vocab)
end

function tokenize_char_level(strs; max_length = 4000)::Matrix{Char}
	truncated = [length(s) > max_length ? SubString(s, 1, max_length) : s for s in strs]
	char_matrix = fill(' ', max_length, length(truncated))
	for i in eachindex(truncated)
		char_matrix[1:length(truncated[i]), i] .= collect(truncated[i])
	end
	return char_matrix
end

function download_and_extract_dataset(; download_folder = "download", dataset_folder = "lra_release", url = "https://storage.googleapis.com/long-range-arena/lra_release.gz")
	gz_file_path = joinpath(download_folder, "$dataset_folder.gz")
	file_path = joinpath(download_folder, dataset_folder)
	println("Start downloading...")
	
	last_update = 0
	function download_progress_callback(total, now, num_updates=10)
		step = total / num_updates
		if now - last_update >= step || now == total
			val = round(now / total * 100, digits=2)
			if isnan(val)
				return
			end
			println("Download progress: $val%")
			last_update = now
		end
	end

	Downloads.download(url, gz_file_path, progress = download_progress_callback)
	println("Download completed...")
	println("Start extraction...")
	open(GzipDecompressorStream, gz_file_path) do io
		Tar.extract(io, file_path)
	end
	println("Extraction completed.")
	#rm(gz_file_path)
end


function get_lra_retrieval(; seq_len = 4000, data_to_use_percent = 1, data_folder = "data/lra_retrieval")

	if !isdir("$data_folder")

		println("No serialized data found")

		if !isdir("download/lra_release")
			println("Dataset file not found.")
			download_and_extract_dataset()
		end

		println("Start loading dataset.")

		mkpath(data_folder)

		tsv_data_folder = "download/lra_release/lra_release/lra_release/tsv_data"

		num_rows = count(_ -> true, eachline("$tsv_data_folder/new_aan_pairs.train.tsv"))
		train_text_1 = fill(' ', seq_len, num_rows)
		train_text_2 = fill(' ', seq_len, num_rows)
		trainY = Vector{Float32}(undef, num_rows)

		open("$tsv_data_folder/new_aan_pairs.train.tsv") do file

			for (i, line) in enumerate(eachline(file))

				parts = split(line, '\t')
				trainY[i] = parse(Float32, parts[1])

				tokenized_text_1 = tokenize_char_level([parts[4]]; max_length=seq_len)  # Character matrix (1 row)
				tokenized_text_2 = tokenize_char_level([parts[5]]; max_length=seq_len)

				train_text_1[1:length(tokenized_text_1), i] = tokenized_text_1
				train_text_2[1:length(tokenized_text_2), i] = tokenized_text_2

			end
		end

		println("Training dataset loaded")

		validation_data = readdlm("$tsv_data_folder/new_aan_pairs.eval.tsv", '\t', header = false)

		validationY = validation_data[:, 1]
		validation_text_1 = tokenize_char_level(validation_data[:, 4]; max_length=seq_len)
		validation_text_2 = tokenize_char_level(validation_data[:, 5]; max_length=seq_len)

		println("Validation dataset loaded.")

		println("Start building vocab.")

		vocab = get_vocab(
			hcat(train_text_1, train_text_2, validation_text_1, validation_text_2);
			min_freq = 1)

		println("Vocab built.")

		vocab_dict = Dict(c => i for (i, c) in enumerate(vocab))

		validation_text_1 = map(c -> vocab_dict[c], validation_text_1)
		validation_text_2 = map(c -> vocab_dict[c], validation_text_2)
		train_text_1 = map(c -> vocab_dict[c], train_text_1)
		train_text_2 = map(c -> vocab_dict[c], train_text_2)

		println("Start serializing data")

		Serialization.serialize("$data_folder/vocab.jls", vocab)
		Serialization.serialize("$data_folder/train_text_1.jls", train_text_1)
		Serialization.serialize("$data_folder/train_text_2.jls", train_text_2)
		Serialization.serialize("$data_folder/trainY.jls", trainY)
		Serialization.serialize("$data_folder/validation_text_1.jls", validation_text_1)
		Serialization.serialize("$data_folder/validation_text_2.jls", validation_text_2)
		Serialization.serialize("$data_folder/validationY.jls", validationY)

		println("Serialization done")

	else
		vocab = Serialization.deserialize("$data_folder/vocab.jls")
		train_text_1 = Serialization.deserialize("$data_folder/train_text_1.jls")
		train_text_2 = Serialization.deserialize("$data_folder/train_text_2.jls")
		trainY = Serialization.deserialize("$data_folder/trainY.jls")
		validation_text_1 = Serialization.deserialize("$data_folder/validation_text_1.jls")
		validation_text_2 = Serialization.deserialize("$data_folder/validation_text_2.jls")
		validationY = Serialization.deserialize("$data_folder/validationY.jls")
	end

	train_split = floor(Int, size(train_text_1, 2) * data_to_use_percent)
	validation_split = floor(Int, size(validation_text_1, 2) * data_to_use_percent)

	("Data loaded")

	return (
		vocab,
		train_text_1[:, 1:train_split],
		train_text_2[:, 1:train_split],
		trainY[1:train_split],
		validation_text_1[:, 1:validation_split],
		validation_text_2[:, 1:validation_split],
		validationY[1:validation_split],
	)

end
