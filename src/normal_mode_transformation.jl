
struct NormalModeTransformation{T}
    U::Matrix{T}
    tmp::Vector{T}
    function NormalModeTransformation{T}(n) where {T}
        # Real and imaginary parts of the discrete Fourier transform matrix. 
        U = [
            k > n ÷ 2 ?
            sin(2π * j * k / n) : cos(2π * j * k / n)
            for j in 0:n-1, k in 0:n-1
        ]

        for col in eachcol(U)
            normalize!(col)
        end

        return new{T}(U, zeros(n))
    end
end

function transform_to_normal_modes!(A::RingPolymerArray, transform::NormalModeTransformation)
    transform!(A, transpose(transform.U), transform.tmp)
end

function transform_from_normal_modes!(A::RingPolymerArray, transform::NormalModeTransformation)
    transform!(A, transform.U, transform.tmp)
end

function transform!(A::RingPolymerArray, U::AbstractMatrix, tmp::AbstractVector)
    @views for i in quantumindices(A)
        for j in axes(A, 1)
            mul!(tmp, U, A[j,i,:])
            copy!(A[j,i,:], tmp)
        end
    end
end
