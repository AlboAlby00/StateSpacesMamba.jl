using Flux, Optimisers, ProgressMeter
using MLDatasets
using CUDA
using Statistics
CUDA.allowscalar(false)

# hyperparameters
n_classes = 10
batch_size = 128
test_batch_size = 32
hidden_size = 256
epochs = 1000
device = gpu_device()

# create split mnist dataloader

# train loader
train_images, train_labels = MNIST(split = :train)[:]
n_train_images = size(train_images)[end]
train_sequences = reshape(train_images, 1, 28 * 28, n_train_images)
train_labels = repeat(train_labels', outer = (28 * 28, 1))
train_loader = Flux.DataLoader((train_sequences, train_labels), batchsize = batch_size, shuffle = true, partial = false)

for (x, y) in train_loader
	@assert(size(x) == (1, 28 * 28, batch_size))
end

# test loader
test_images, test_labels = MNIST(split=:test)[:]
n_test_images = size(test_images)[end]
test_sequences = reshape(test_images, 1, 28 * 28, n_test_images)
test_labels = repeat(test_labels', outer=(28 * 28, 1))
test_loader = Flux.DataLoader((test_sequences, test_labels), batchsize=test_batch_size, partial = false)

for (x, y) in test_loader
    @assert(size(x) == (1, 28 * 28, test_batch_size))
end

model = Chain(
	LSTM(1 => hidden_size),
	Dense(hidden_size => hidden_size, relu),
	Dense(hidden_size => n_classes, identity),
) |> device

opt_state = Flux.setup(Flux.Adam(0.001), model)

losses = []
@showprogress "Training progress..." for epoch in 1:epochs # Training loop
	Flux.reset!(model)
	train_progress = Progress(length(train_loader))
	for (x, y) in train_loader
		x, y = x |> device, y |> device
		loss, grads = Flux.withgradient(model) do m
			logits = m(x)
			onehot_y = Flux.onehotbatch(y, 0:n_classes-1)
			Flux.Losses.logitcrossentropy(logits, onehot_y)
		end
		Flux.update!(opt_state, model, grads[1])
		push!(losses, loss)
		next!(train_progress; showvalues = [("Epoch ", epoch), ("Loss ", loss)])
	end
	accuracy_list = []
	test_progress = Progress(length(test_loader))
	for (x, y) in test_loader
        x = x |> device
        logits = Array(model(x)) # move back to CPU
        predictions = dropdims(argmax(logits, dims=1), dims=1)
		# convert CartesianIndex matrix to int matrix
        predictions = reshape([predictions[i, j][1] for i in 1:size(logits, 2) for j in 1:size(logits, 3)], size(y)) 
        accuracy = sum(predictions .== y) / length(predictions)
        push!(accuracy_list, accuracy)
		next!(test_progress; showvalues=[("Epoch ", epoch), ("Accuracy ", mean(accuracy_list))])
    end
end

