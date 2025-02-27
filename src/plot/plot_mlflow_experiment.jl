using Plots
using Plots.PlotMeasures
using Statistics
using YAML
using DelimitedFiles

mlflow_experiment_folder = "mamba_all_dropout_search"
mkpath("images")

colors = [:blue, :red, :green, :brown, :purple, :orange, :pink, :cyan, :magenta, :black, :white, :gray, :lime]

combined_plot = plot(
	title = "Loss Curve Comparison",
	xlabel = "Steps",
	ylabel = "Loss",
	legend = :topright,
	linewidth = 2,
	ylims = (1.0, 2.0),
	size = (1400, 800),
	left_margin = 15mm,
	right_margin = 20mm,
	bottom_margin = 10mm,
)

subdirs = filter(isdir, readdir("mlruns/$(mlflow_experiment_folder)", join = true))

all_train_losses = Dict()
all_test_losses = Dict()
min_test_losses = Dict()

for run_folder in subdirs
	metadata = YAML.load_file(joinpath(run_folder, "meta.yaml"))
	run_name = metadata["run_name"]

	combination_name = replace(run_name, r"iteration=\d+, " => "")

	train_losses_lines = readdlm(joinpath(run_folder, "metrics/train loss"), ' ')
	test_losses_lines = readdlm(joinpath(run_folder, "metrics/validation loss"), ' ')

	train_losses = train_losses_lines[:, 2]
	test_losses = test_losses_lines[:, 2]

	if !haskey(all_train_losses, combination_name)
		all_train_losses[combination_name] = []
		all_test_losses[combination_name] = []
		min_test_losses[combination_name] = []
	end

	push!(all_train_losses[combination_name], train_losses)
	push!(all_test_losses[combination_name], test_losses)
	push!(min_test_losses[combination_name], minimum(test_losses))
end

for (color_index, combination_name) in enumerate(keys(all_train_losses))

	# Concatenate all training and test losses for each combination
	train_loss_matrix = hcat(all_train_losses[combination_name]...)
	test_loss_matrix = hcat(all_test_losses[combination_name]...)

	mean_train_loss = mean(train_loss_matrix, dims = 2)
	mean_test_loss = mean(test_loss_matrix, dims = 2)
	std_test_loss = std(test_loss_matrix, dims = 2)

	conf_test = 1.96 * std_test_loss / sqrt(size(test_loss_matrix, 2))
	mean_min_test_loss = mean(min_test_losses[combination_name])

	println("Combination: $(combination_name)")
	println("Mean of minimum validation loss across runs: $(mean_min_test_loss)")
	println("----------------------------------------")

	# Plot the training and validation losses
	plot!(combined_plot, 1:length(mean_train_loss), mean_train_loss, label = "$(combination_name) Training Loss", linestyle = :solid, color = colors[color_index])

	test_steps = range(1, length(mean_train_loss), length = length(mean_test_loss))
	plot!(combined_plot, test_steps, mean_test_loss, ribbon = conf_test, label = "$(combination_name) Validation Loss", linestyle = :dash, color = colors[color_index])

	# Increment color index for the next plot
	color_index += 1
	if color_index > length(colors)
		color_index = 1  # Loop back to the first color if there are more runs than colors
	end

end

savefig(combined_plot, "images/mlflow_experiments/$(mlflow_experiment_folder)_loss_curve.png")

