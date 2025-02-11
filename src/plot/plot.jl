using DelimitedFiles
using Plots

experiments = ["small_mamba_with_both_dropout", "small_mamba", "small_mamba_with_standard_dropout", "small_mamba_with_ssm_dropout"]
mkpath("images")

colors = [:blue, :red, :green, :brown]  # One color per model

combined_plot = plot(
	title = "Loss Curve Comparison",
	xlabel = "Steps",
	ylabel = "Loss",
	legend = :topright,
	linewidth = 2,
	ylims = (1.0, 2.0),
	size = (1000, 800) 
)

for (i, experiment) in enumerate(experiments)
	train_losses = vec(readdlm("saved_csv/$(experiment)/train_losses.csv", ','))
	test_losses = vec(readdlm("saved_csv/$(experiment)/test_losses.csv", ','))

	plot!(combined_plot, 1:length(train_losses), train_losses, label = "$(experiment) Training Loss", linestyle = :solid, color = colors[i])
	plot!(combined_plot, range(1, length(train_losses), length = length(test_losses)), test_losses, label = "$(experiment) Test Loss", linestyle = :dash, color = colors[i])

	individual_plot = plot(
		1:length(train_losses), train_losses, label = "Training Loss", xlabel = "Steps", ylabel = "Loss",
		title = "Loss Curve - $(experiment)", linewidth = 2, legend = :topright, ylims = (1.0, 2.0), color = colors[i],
	)
	plot!(range(1, length(train_losses), length = length(test_losses)), test_losses, label = "Test Loss", linewidth = 2, linestyle = :dash, color = colors[i])

	savefig(individual_plot, "images/$(experiment).png")
end

savefig(combined_plot, "images/loss_curve_comparison.png")
