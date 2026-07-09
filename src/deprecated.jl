@deprecate whichcell(x, mesh) findcell(x, mesh)

Base.@deprecate_binding Interpolation Basis
Base.@deprecate_binding InterpolationWeight BasisWeight
Base.@deprecate_binding InterpolationWeightArray BasisWeightArray
Base.@deprecate_binding ColorPartition ThreadPartition

@deprecate generate_interpolation_weights(args...; kwargs...) generate_basis_weights(args...; kwargs...)
@deprecate initial_neighboringnodes(args...) initial_supportnodes(args...)
@deprecate neighboringnodes_storage(bw::BasisWeight) supportnodes_storage(bw)
@deprecate colorgroups(args...) threadsafe_groups(args...)
@deprecate particle_indices_in(args...) particle_indices(args...)

@deprecate interpolation(bw::BasisWeight) (basis(bw)::Basis)
@deprecate interpolation(weights::BasisWeightArray) (basis(weights)::Basis)
@deprecate cellshape(bw::BasisWeight) (basis(bw)::Shape)
@deprecate cellshape(weights::BasisWeightArray) (basis(weights)::Shape)

@deprecate neighboringnodes(bw::BasisWeight) supportnodes(bw)
@deprecate neighboringnodes(bw::BasisWeight, domain) supportnodes(bw, domain)
@deprecate neighboringnodes(basis::Basis, pt, mesh::AbstractMesh) supportnodes(basis, pt, mesh)
@deprecate neighboringnodes(x::Vec, r::Real, mesh::CartesianMesh) supportnodes(x, r, mesh)
