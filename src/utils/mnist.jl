using MLDatasets

function get_mnist_dataloaders(device; input_dim = 1, train_batch_size=64, validation_batch_size=64)

    train_images, train_labels = MNIST(split=:train)[:]
    n_train_images = size(train_images)[end]
    train_sequences = Float32.(reshape(train_images, input_dim, 28 * 28 ÷ input_dim, n_train_images)) # shape is (d, l, b) d -> feature dimension, l -> sequence length, b -> batch size
    #train_labels = Int32.(repeat(train_labels', outer=(28 * 28 ÷ input_dim, 1)))
    train_loader = Flux.DataLoader((train_sequences, train_labels), batchsize=train_batch_size, shuffle=true, partial=false) |> device

    for (x, y) in train_loader
        @assert size(x) == (input_dim, 28 * 28 / input_dim, train_batch_size)
    end

    test_images, test_labels = MNIST(split=:test)[:]
    n_test_images = size(test_images)[end]
    test_sequences = Float32.(reshape(test_images, input_dim, 28 * 28 ÷ input_dim, n_test_images))
    #test_labels = Int32.(repeat(test_labels', outer=(28 * 28 ÷ input_dim, 1)))
    test_loader = Flux.DataLoader((test_sequences, test_labels), batchsize=validation_batch_size, partial=false) |> device

    for (x, y) in test_loader
        @assert size(x) == (input_dim, 28 * 28 / input_dim, validation_batch_size)
    end

    return train_loader, test_loader
end
