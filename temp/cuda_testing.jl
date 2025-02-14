using Test
using BenchmarkTools
using CUDA
using Plots

CUDA.allowscalar(true)

K = 8
BLOCKS = 4
SEQLEN = K * BLOCKS

x = 1:SEQLEN
y = zeros(SEQLEN)

function seq_cumsum(x)
	y = []
	h = 0
	for k in eachindex(x)
		h = h + x[k]
		push!(y, h)
	end
	return h, y
end

h_, y_ = seq_cumsum(x)

# @btime seq_cumsum($x)

# bar(x, y_, xlabel="Index", ylabel="Value", title="Bar Plot", legend=false)

op = (x, y) -> x .+ y

x_d = CuArray(1:SEQLEN)
y_d = CUDA.zeros(SEQLEN)
function cuda_cumsum_1!(x, y)
	CUDA.scan!(+, y, x; dims = 1)
end

#cuda_cumsum_1!(x_d, y_d)
#@assert y_ == Array(y_d)

function hillis_steele_scan!(x, op, tid, kid)

	offset = 1
	while offset < blockDim().x
		temp = x[tid]  # Preserve current value
		if tid > offset
			temp = op(temp, x[tid-offset])
		end
		x[tid] = temp  # Update shared memory
		offset *= 2
		sync_threads()
	end

end

function cuda_cumsum_for_each_block!(
	X::CuDeviceVector{T}, Y::CuDeviceVector{T},
	H0::CuDeviceVector{T}, H::CuDeviceVector{T}) where T

	kid = (blockIdx().x - 1) * blockDim().x
	tid = threadIdx().x

	x = CuDynamicSharedArray(T, blockDim().x)
	@inbounds x[tid] = X[kid+tid]


	sync_threads()

	hillis_steele_scan!(x, +, tid, kid)

	@inbounds Y[kid+tid] = x[tid]
	@inbounds H[kid] = x[end]

	sync_threads()

	return nothing
end

function cuda_cumsum!(X, Y, k, blocks)
	H = CUDA.zeros(Float32, 2, blocks)
	@cuda threads = k blocks = blocks shmem = (k * sizeof(Float32)) cuda_cumsum_for_each_block!(X, Y, H[1,:], H[1,:])
    println(H)
    H[2,2:end] = cumsum(H[1,:]; dims=1)[1:end-1]
    println(H)
	@cuda threads = k blocks = blocks shmem = (k * sizeof(Float32)) cuda_cumsum_for_each_block!(X, Y, H[2,:], H[2,:])

	return nothing
end


x_d = Float32.(CuArray(1:SEQLEN))
y_d = CUDA.zeros(Float32, SEQLEN)

cuda_cumsum!(x_d, y_d, K, BLOCKS)

#@btime cumsum(x_d)
#@btime cuda_cumsum!(x_d, y_d, K, BLOCKS)
cuda_cumsum!(x_d, y_d, K, BLOCKS)
#= println(y_d[1:20])
println(cumsum(x_d)[1:20]) =#
@assert cumsum(x_d) == y_d
