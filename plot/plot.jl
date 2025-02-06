using DelimitedFiles
using Plots

# List of models to compare
models = ["mamba", "transformer"]

# Ensure the "images" directory exists
mkpath("images")

# Initialize a plot for the combined losses
combined_plot = plot(title="Loss Curve Comparison", xlabel="Steps", ylabel="Loss", legend=:topright, linewidth=2)

# Loop through each model
for model in models
    # Read training and test losses from CSV files
    train_losses = vec(readdlm("saved_csv/train_losses_$(model).csv", ','))
    test_losses = vec(readdlm("saved_csv/test_losses_$(model).csv", ','))

    # Plot training losses
    plot!(combined_plot, 1:length(train_losses), train_losses, 
          label="$(model) Training Loss", 
          linestyle=:solid)

    # Plot test losses
    plot!(combined_plot, range(1, length(train_losses), length=length(test_losses)), test_losses, 
          label="$(model) Test Loss", 
          linestyle=:dash)

    # Save individual model plots
    individual_plot = plot(1:length(train_losses), train_losses, 
                           label="Training Loss", 
                           xlabel="Steps", ylabel="Loss", 
                           title="Loss Curve - $(model)", 
                           linewidth=2, legend=:topright)
    plot!(range(1, length(train_losses), length=length(test_losses)), test_losses, 
          label="Test Loss", 
          linewidth=2, linestyle=:dash)
    savefig(individual_plot, "images/loss_curve_$(model).png")
end

# Save the combined plot
savefig(combined_plot, "images/loss_curve_comparison.png")

# Display the combined plot
display(combined_plot)