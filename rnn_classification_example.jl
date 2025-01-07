using DataLoaders
using Flux, Optimisers, ProgressMeter
using MLDatasets
using CUDA
CUDA.allowscalar(false)

# hyperparameters
n_classes = 10
batch_size = 32
hidden_size = 128
epochs = 5

# create split mnist dataloader

# train loader
train_images, train_labels = MNIST(split=:train)[:]
n_train_images = size(train_images)[end]
train_sequences = reshape(train_images, 1, 28*28, n_train_images)
train_labels = repeat(train_labels', outer=(28*28, 1))
train_loader = DataLoader((train_sequences, train_labels), batch_size)

for (batch_data, batch_label) in train_loader
    @assert size(batch_data) == (1, 28*28,batch_size)
end


# test loader
test_images, test_labels = MNIST(split=:test)[:]
n_test_images = size(test_images)[end]
test_sequences = reshape(test_images, 28*28, n_test_images)
test_loader = DataLoader((test_sequences, test_labels), batch_size)

model = Chain(
    x -> reshape(x, 1, 28*28, batch_size),
    LSTM(1 => hidden_size),
    Dense(hidden_size => n_classes, identity)
)

opt_state = Flux.setup(Flux.Adam(0.001), model) 

losses = []
@showprogress "Training progress..." for epoch in 1:epochs # Training loop
    Flux.reset!(model)
    progress = Progress(length(train_loader))
    for (x, y) in train_loader
        loss, grads = Flux.withgradient(model) do m
            logits = m(x)
            onehot_y = Flux.onehotbatch(y, 0:n_classes-1)
            Flux.Losses.logitcrossentropy(logits, onehot_y)
        end
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)
        next!(progress; showvalues = [("Epoch ",epoch), ("Loss ",loss)])
    end
end

