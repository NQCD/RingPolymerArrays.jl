"""
    RingPolymerArrays

# Requirements

* DiffEq usage
* Variable number of replicas for different dofs/atoms
* Normal mode transformation

# Temporary decisions

* Assume `(ndofs, natoms)` size for each bead.
* Number of replicas must be `nbeads` or `1`.

"""
module RingPolymerArrays

export RingPolymerArray
export eachbead
export eachdof

using LinearAlgebra: normalize!, mul!
using OrderedCollections: OrderedDict

struct IndexMapping
    classical::OrderedDict{Int,Int}
    quantum::OrderedDict{Int,Int}
end

function IndexMapping(quantum::Vector{Int}, classical::Vector{Int})
    classical_map = OrderedDict(classical .=> 1:length(classical))
    quantum_map = OrderedDict(quantum .=> 1:length(quantum))
    return IndexMapping(classical_map, quantum_map)
end

struct RingPolymerArray{T} <: AbstractArray{T,3}
    classical_atoms::Matrix{T}
    quantum_atoms::Vector{Matrix{T}}
    index_map::IndexMapping
    dims::Dims{3}
end

function RingPolymerArray(A::AbstractArray{T,3}; classical=Int[]) where {T}
    B = RingPolymerArray{T}(undef, size(A); classical)
    copy!(B, A)
    return B
end

function RingPolymerArray(A::AbstractMatrix{T}; nbeads, classical=Int[]) where {T}
    B = RingPolymerArray{T}(undef, (size(A)..., nbeads); classical)
    for bead in eachbead(B)
        copy!(bead, A)
    end
    return B
end

function RingPolymerArray{T}(::UndefInitializer, dims::Dims{3}; classical::AbstractVector{<:Integer}) where {T}
    quantum = find_quantum_indices(dims[2], classical)
    index_map = IndexMapping(quantum, classical)
    classical_atoms = Matrix{T}(undef, dims[1], length(classical))
    quantum_atoms = [Matrix{T}(undef, dims[1], length(quantum)) for _ in 1:dims[3]]
    return RingPolymerArray{T}(classical_atoms, quantum_atoms, index_map, dims)
end

function find_quantum_indices(natoms::Integer, classical::AbstractVector{<:Integer})
    quantum = collect(1:natoms)
    setdiff!(quantum, classical)
    return quantum
end

function Base.similar(A::RingPolymerArray, ::Type{T}, dims::Dims) where {T}
    return RingPolymerArray{T}(undef, dims; classical=collect(classicalindices(A)))
end

Base.size(A::RingPolymerArray) = A.dims

function Base.getindex(A::RingPolymerArray, i, j, k)
    j in classicalindices(A) ? get_classical_index(A, i, j, k) : get_quantum_index(A, i, j, k)
end

get_classical_index(A::RingPolymerArray, i, j, k) = A.classical_atoms[i, A.index_map.classical[j]]
get_quantum_index(A::RingPolymerArray, i, j, k) = A.quantum_atoms[k][i, A.index_map.quantum[j]]

function Base.setindex!(A::RingPolymerArray, v, i, j, k)
    j in classicalindices(A) ? set_classical_index!(A, v, i, j, k) : set_quantum_index!(A, v, i, j, k)
end

function set_classical_index!(A::RingPolymerArray, v, i, j, k)
    k == 1 || return # Do nothing if trying to set classical atoms for other beads
    setindex!(A.classical_atoms, v, i, A.index_map.classical[j])
end

function set_quantum_index!(A::RingPolymerArray, v, i, j, k)
    setindex!(A.quantum_atoms[k], v, i, A.index_map.quantum[j])
end

quantumindices(A::RingPolymerArray) = keys(A.index_map.quantum)
classicalindices(A::RingPolymerArray) = keys(A.index_map.classical)

eachbead(A::RingPolymerArray) = (view(A, :, :, i) for i in axes(A, 3))
eachdof(A::RingPolymerArray) = (view(A, i, j, :) for i in axes(A, 1), j in axes(A, 2))

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

end # module
