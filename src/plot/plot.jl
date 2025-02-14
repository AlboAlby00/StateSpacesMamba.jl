using DelimitedFiles
using Plots
using Statistics

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

    subdirs = readdir("saved_csv/$(experiment)", join=true)
    
    all_train_losses = []
    all_test_losses = []
    min_test_losses = []  
    
    for subdir in subdirs
        train_losses = vec(readdlm(joinpath(subdir, "train_losses.csv"), ','))
        test_losses = vec(readdlm(joinpath(subdir, "test_losses.csv"), ','))
        
        push!(all_train_losses, train_losses)
        push!(all_test_losses, test_losses)
        push!(min_test_losses, minimum(test_losses))
    end
    
    # Convert to a matrix where each row is a run
    train_loss_matrix = hcat(all_train_losses...)
    test_loss_matrix = hcat(all_test_losses...)
    
    mean_train_loss = mean(train_loss_matrix, dims=2)
    
    mean_test_loss = mean(test_loss_matrix, dims=2)
    std_test_loss = std(test_loss_matrix, dims=2)
    
    # Compute 95% confidence interval for test loss
    conf_test = 1.96 * std_test_loss / sqrt(size(test_loss_matrix, 2))
    
    # Compute the mean of the minimum test loss values across all runs
    mean_min_test_loss = mean(min_test_losses)
    
    println("Experiment: $(experiment)")
    println("Mean of minimum test loss across runs: $(mean_min_test_loss)")
    println("----------------------------------------")
    
    plot!(combined_plot, 1:length(mean_train_loss), mean_train_loss, label = "$(experiment) Training Loss", linestyle = :solid, color = colors[i])
    
    test_steps = range(1, length(mean_train_loss), length = length(mean_test_loss))
    plot!(combined_plot, test_steps, mean_test_loss, ribbon=conf_test, label = "$(experiment) Test Loss", linestyle = :dash, color = colors[i])
    
    individual_plot = plot(
        1:length(mean_train_loss), mean_train_loss, label = "Training Loss", xlabel = "Steps", ylabel = "Loss",
        title = "Loss Curve - $(experiment)", linewidth = 2, legend = :topright, ylims = (1.0, 2.0), color = colors[i],
    )
    plot!(test_steps, mean_test_loss, ribbon=conf_test, label = "Test Loss", linewidth = 2, linestyle = :dash, color = colors[i])
    
    savefig(individual_plot, "images/$(experiment).png")
end

savefig(combined_plot, "images/loss_curve_comparison.png")