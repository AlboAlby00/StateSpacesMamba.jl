using MLDatasets
using Images  # For displaying images

# Load MNIST dataset
train_images, train_labels = MNIST.traindata()
test_images, test_labels = MNIST.testdata()

print("hello world")