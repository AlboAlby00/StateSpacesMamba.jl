using StatsBase

function get_tiny_shakespeare(; seq_len=64, test_percent=0.2, data_to_use_percent=1)
    isfile("download/input.txt") || download(
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        "download/input.txt",
    )

    text = String(read("download/input.txt"))

    text = replace(text, r"\r?\n" => " ")

    alphabet = [unique(text)..., '_']
    stop = alphabet[end]

    text = text[1:floor(Int, length(text)*data_to_use_percent)]

    B = (length(text)-1) ÷ seq_len
    Xs = reshape(collect(text)[1:B*seq_len], seq_len, B)
    Ys = reshape(collect(text)[2:B*seq_len+1], seq_len, B)

    Xs[1,:] .= stop

    Xs = map(c -> Int32(findfirst(==(c), alphabet)), Xs)
    Ys = Flux.onehotbatch(Ys, alphabet)

    numbatch = size(Xs, 2)
    split = floor(Int, (1-test_percent) * numbatch)

    trainX, trainY = Xs[:,1:split],       Ys[:,:,1:split]
    testX,  testY =  Xs[:,(split+1):end], Ys[:,:,(split+1):end]

    return (alphabet, trainX, trainY, testX, testY)
end

function generate(model, alphabet, seed, outlen, seqlen)
    if isempty(seed)
        seed = "_"
    end
    x = map(c -> findfirst(==(c), alphabet)::Int64, collect(seed))
    while length(x) < outlen
        tail = x[max(1, end-seqlen+1):end]
        tail = reshape(tail, length(tail), 1)
        y = model(tail |> device) |> cpu
        p = softmax(y[:,end,1])
        j = sample(1:length(alphabet), Weights(p))
        push!(x, j)
    end
    String(map(j -> alphabet[j], x))
end