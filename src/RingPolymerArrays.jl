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

using LinearAlgebra: normalize!, mul!
using OrderedCollections: OrderedDict

include("ring_polymer_array.jl")
export RingPolymerArray
export eachbead
export eachdof

include("normal_mode_transformation.jl")

end # module
