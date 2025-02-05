models = ["mamba", "transformer"]
mkpath("images")  # Ensure directory exists

for model in models

    train_losses = vec(readdlm("saved_csv/train_losses_$(model).csv", ','))
    test_losses = vec(readdlm("saved_csv/test_losses_$(model).csv", ','))

    p = plot(1:length(train_losses), train_losses, label="Training Loss", xlabel="Steps", ylabel="Loss", title="Loss Curve - $(model)", lw=2)
    plot!(range(1, length(train_losses), length=length(test_losses)), test_losses, label="Test Loss", lw=2, linestyle=:dash)

    display(p)

    savefig("images/train_losses_$(model).png")
end