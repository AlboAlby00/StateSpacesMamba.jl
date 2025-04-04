using Pkg
Pkg.activate(".")

using Plots
using Plots.PlotMeasures
using Statistics
using YAML
using DelimitedFiles
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
	"--name", "-n"
		help = "experiment name that you want to plot"
		arg_type = String
		required = true
	"--train"
		help = "Flag for setting the 'ex' parameter to true"
		arg_type = Bool
		default = true
end
args = parse_args(s)


mlflow_experiment_folder = "mamba__search"
mkpath("images")

colors = [:blue, :red, :green, :brown, :purple, :orange, :pink, :cyan, :magenta, :black, :white, :gray, :lime]

combined_plot = plot(
	title = "Loss Curve Comparison",
	xlabel = "Steps",
	ylabel = "Loss",
	legend = :topright,
	linewidth = 2,
	ylims = (0.0, 0.2),
	size = (1400, 800),
	left_margin = 15mm,
	right_margin = 20mm,
	bottom_margin = 10mm,
)

subdirs = filter(isdir, readdir("mlruns/$(args["name"])", join = true))

all_train_losses = Dict()
all_validation_losses = Dict()
min_validation_losses = Dict()

for run_folder in subdirs
	metadata = YAML.load_file(joinpath(run_folder, "meta.yaml"))
	run_name = metadata["run_name"]

	combination_name = replace(run_name, r"iteration=\d+, " => "")

	train_losses_lines = readdlm(joinpath(run_folder, "metrics/train loss"), ' ')
	validation_losses_lines = readdlm(joinpath(run_folder, "metrics/validation loss"), ' ')

	train_losses = train_losses_lines[:, 2]
	validation_losses = validation_losses_lines[:, 2]

	if !haskey(all_train_losses, combination_name)
		all_train_losses[combination_name] = []
		all_validation_losses[combination_name] = []
		min_validation_losses[combination_name] = []
	end

	push!(all_train_losses[combination_name], train_losses)
	push!(all_validation_losses[combination_name], validation_losses)
	push!(min_validation_losses[combination_name], minimum(validation_losses))
end

for (color_index, combination_name) in enumerate(keys(all_train_losses))

	# Concatenate all training and test losses for each combination
	train_loss_matrix = hcat(all_train_losses[combination_name]...)
	validation_loss_matrix = hcat(all_validation_losses[combination_name]...)

	mean_train_loss = mean(train_loss_matrix, dims = 2)
	mean_validation_loss = mean(validation_loss_matrix, dims = 2)
	std_validation_loss = std(validation_loss_matrix, dims = 2)

	conf_test = 1.96 * std_validation_loss / sqrt(size(validation_loss_matrix, 2))
	mean_min_validation_loss = mean(min_validation_losses[combination_name])

	println("Combination: $(combination_name)")
	println("Mean of minimum validation loss across runs: $(mean_min_validation_loss)")
	println("----------------------------------------")

	# Plot the training and validation losses
	if args["train"]
		plot!(combined_plot, 1:length(mean_train_loss), mean_train_loss, label = "$(combination_name) Training Loss", linestyle = :solid, color = colors[color_index])
	end

	validation_steps = range(1, length(mean_train_loss), length = length(mean_validation_loss))

	validation_steps = range(1, length(mean_validation_loss), length = length(mean_validation_loss))
	plot!(combined_plot, validation_steps, mean_validation_loss, ribbon = conf_test, label = "$(combination_name) Validation Loss", linestyle = :dash, color = colors[color_index])	

	# Increment color index for the next plot
	color_index += 1
	if color_index > length(colors)
		color_index = 1  # Loop back to the first color if there are more runs than colors
	end

end

mkpath("images/mlflow_experiments")
savefig(combined_plot, "images/mlflow_experiments/$(args["name"])_loss_curve.png")
