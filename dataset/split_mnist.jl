using MLDatasets
using Images
using PyPlot

pygui(true)  # Enable interactive plotting for PyPlot

train_images, train_labels = MNIST(split=:train)[:]
test_images, test_labels = MNIST(split=:test)[:]

for i in 1:10
    label = train_labels[i]
    
    imshow(train_images[:, :, i])
    title("Label: $label")          
    show()                          
    sleep(1)                        
end
