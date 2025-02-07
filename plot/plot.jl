using DelimitedFiles
using Plots

models = ["transformer_with_dropout", "transformer", "mamba"]
mkpath("images")

colors = [:blue, :red, :green]  # One color per model

combined_plot = plot(
	title = "Loss Curve Comparison",
	xlabel = "Steps",
	ylabel = "Loss",
	legend = :topright,
	linewidth = 2,
	ylims = (1.0, 2.0),
)

for (i, model) in enumerate(models)
	train_losses = vec(readdlm("saved_csv/train_losses_$(model).csv", ','))
	test_losses = vec(readdlm("saved_csv/test_losses_$(model).csv", ','))

	plot!(combined_plot, 1:length(train_losses), train_losses, label = "$(model) Training Loss", linestyle = :solid, color = colors[i])
	plot!(combined_plot, range(1, length(train_losses), length = length(test_losses)), test_losses, label = "$(model) Test Loss", linestyle = :dash, color = colors[i])

	individual_plot = plot(
		1:length(train_losses), train_losses, label = "Training Loss", xlabel = "Steps", ylabel = "Loss",
		title = "Loss Curve - $(model)", linewidth = 2, legend = :topright, ylims = (1.0, 2.0), color = colors[i],
	)
	plot!(range(1, length(train_losses), length = length(test_losses)), test_losses, label = "Test Loss", linewidth = 2, linestyle = :dash, color = colors[i])

	savefig(individual_plot, "images/loss_curve_$(model).png")
end

savefig(combined_plot, "images/loss_curve_comparison.png")
