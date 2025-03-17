# Mamba SSM with SSM dropout
![Alt text](media/mamba_diagram.png)


This repository contains:  
- **First Mamba SSM implementation in Julia** (as far as I know) using `Flux.jl` library.   
- **Optimized CUDA scan** minimizing global memory access, implemented with `CUDA.jl` library.
- **Scripts** scripts to train the Mamba architecture on the following tasks:
    - Text generation on the tiny_shakespeare dataset with character level tokenization.
    - Classification on the sequential MNIST dataset (each image is treated as a sequence of pixels).
    - Retrieval task from the LRA benchmark.
- **Original SSM dropout** effective novel way of applying dropout in the Mamba architecture.

# References
Here is a list of resources and repos from which I have taken inspiration while realizing this project.
- [Original Mamba implementation](https://github.com/state-spaces/mamba) in Pytorch by Albert Gu and Tri Dao
- [mamba-minimal repo](https://github.com/johnma2006/mamba-minimal)
- [The Annotated S4](https://srush.github.io/annotated-s4/#experiments-mnist)
- [Mamba: The Hard Way](https://srush.github.io/annotated-mamba/hard.html)
