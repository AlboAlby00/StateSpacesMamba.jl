using Flux, Optimisers, ProgressMeter
using Statistics, MLDatasets, CUDA, Revise, Plots, DelimitedFiles, BSON, YAML

CUDA.allowscalar(false)

include("src/utils/shakespeare.jl")
include("src/utils/common.jl")
include("src/utils/models.jl")

device = gpu_device()

function train_and_evaluate(experiment, hp)

	alphabet, trainX, trainY, testX, testY = get_tiny_shakespeare(seq_len = hp["seq_len"], data_to_use_percent = hp["data_to_use_percent"])

	train_loader = Flux.DataLoader((trainX, trainY), batchsize = hp["train_batch_size"], shuffle = true, partial = false) |> device
	test_loader = Flux.DataLoader((testX, testY), batchsize = hp["test_batch_size"], shuffle = false, partial = false) |> device

    lr_decay_factor = (hp["final_lr"] / hp["initial_lr"])^(1 / hp["num_epochs"])

	for iteration in 1:hp["num_iterations"]

        set_seed(iteration)

        model = get_model(alphabet, hp) |> f32 |> device
        println("model for experiment '$experiment' has $(count_params(model)) parameters")
        println("start iteration $iteration")
    
        opt = Optimisers.Adam(hp["initial_lr"])
        opt_state = Optimisers.setup(opt, model)

		best_model = nothing
		best_test_loss = Inf

        test_losses = []
        train_losses = []

		for epoch in 1:hp["num_epochs"]

			new_lr = hp["initial_lr"] * lr_decay_factor^(epoch - 1)
			Optimisers.adjust!(opt_state, new_lr)

			Flux.testmode!(model)
			test_progress = Progress(length(test_loader), desc = "Validating Epoch $epoch")
			losses = []
			for (x, y) in test_loader
				y = y |> cpu
				logits = model(x) |> cpu
				test_loss = Flux.Losses.logitcrossentropy(logits, y)
				push!(losses, test_loss)
				next!(test_progress; showvalues = [("Mean Loss", mean(losses))])
			end
			epoch_test_loss = mean(losses)
			push!(test_losses, epoch_test_loss)

			if epoch_test_loss < best_test_loss
				best_test_loss = epoch_test_loss
				best_model = deepcopy(model)
			end

			println(generate(model, alphabet, "The ", 200, hp["seq_len"]))

			Flux.trainmode!(model)
			loss_moving_avg = nothing
			α = 0.05
			train_progress = Progress(length(train_loader), desc = "Training Epoch $epoch")
			for (x, y) in train_loader
				loss, grads = Flux.withgradient(m -> Flux.Losses.logitcrossentropy(m(x), y), model)
				if loss == NaN
					print("NaN value")
					model = best_model
					continue
				end
				Flux.update!(opt_state, model, grads[1])
				if loss_moving_avg === nothing
					loss_moving_avg = loss
				else
					loss_moving_avg = α * loss + (1 - α) * loss_moving_avg
				end
				push!(train_losses, loss)
				next!(train_progress; showvalues = [("Loss", loss_moving_avg)])
			end
		end

		if hp["save_csv"]
			save_losses(experiment, train_losses, test_losses, iteration)
		end

		if hp["save_model"]
			save_model_weights(best_model, experiment, iteration)
		end
	end
end


small_mamba_experiments = ["small_mamba_with_ssm_dropout", "small_mamba", "small_mamba_with_both_dropout"]
mamba_experiments = ["mamba_with_standard_dropout", "mamba_with_ssm_dropout", "mamba", "mamba_with_both_dropout"]
# Run for multiple models
for experiment in ["small_mamba_with_standard_dropout"]
	params = YAML.load_file("experiments/$experiment.yaml")
	train_and_evaluate(experiment, params)
end
