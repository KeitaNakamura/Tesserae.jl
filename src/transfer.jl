using MacroTools

struct TransferEquation
    kind::Symbol
    lhs::Any
    rhs::Any
    op::Symbol
end

is_sum(eq::TransferEquation) = eq.kind === :sum

struct TransferProgram
    equations::Vector{TransferEquation}
    interpolations::Vector{Pair{Symbol, Any}}
end

function split_sum_equations(program::TransferProgram, macroname::String)
    equations = program.equations
    issum = map(is_sum, equations)
    if !allequal(issum) && !issorted(issum; rev=true)
        error("$macroname: Equations without `@∑` must come after those with `@∑`")
    end
    equations[issum], equations[.!issum]
end

# A weight reference like `w[ip]` resolves against the parent weights array:
# through a per-particle column view bound once outside the support loop when
# the scope caches replacements, as `weights.w[ip, p]` inline otherwise. The
# per-particle `BasisWeight` this used to build kept enough live SubArray state
# across the loop to spill GPU registers into local memory, and that spill
# traffic -- not the interpolation -- dominated the transfer kernels.
struct TrailingIndexed
    parent::Any     # the weights array
    trailing::Any   # the particle index
    particles::Any  # particle container, for the position a deferred value needs
    grid::Any       # grid, for its mesh
    window::Any     # support window symbol, mapping a support index to a node
    names::Any      # every weight property the equations reference
    cols::Any       # symbol bound once per particle
    vals::Any       # symbol bound once per support node
end
TrailingIndexed(parent, trailing) = TrailingIndexed(parent, trailing, nothing, nothing, nothing, nothing, nothing, nothing)
function TrailingIndexed(parent, trailing, particles, grid, window, names)
    TrailingIndexed(parent, trailing, particles, grid, window, Tuple(names), gensym(:wcols), gensym(:wvals))
end

struct TransferScope
    bindings::Dict{Any,Any}
    replacements::Union{Nothing,Dict{Any,Vector{Expr}}}
end

function TransferScope(maps::Vector{<: Pair}; cache::Bool=false)
    bindings = Dict{Any,Any}()
    for map in maps
        parent, index = map
        haskey(bindings, index) && error("transfer index `$index` is bound more than once")
        bindings[index] = parent
    end
    replacements = cache ? Dict{Any,Vector{Expr}}(index => Expr[] for index in keys(bindings)) : nothing
    TransferScope(bindings, replacements)
end

uncached(scope::TransferScope) = TransferScope(scope.bindings, nothing)

function cached_replacements(scope::TransferScope, indices...)
    scope.replacements === nothing && error("reference cache is not enabled for this transfer scope")
    exprs = Expr[]
    for index in indices
        haskey(scope.replacements, index) || error("index `$index` is not bound in this transfer scope")
        union!(exprs, scope.replacements[index])
    end
    exprs
end

function resolve_equation(eq::TransferEquation, scope::TransferScope)
    TransferEquation(eq.kind, resolve_refs(eq.lhs, scope), resolve_refs(eq.rhs, scope), eq.op)
end

function resolve_sum_equations(equations::Vector{TransferEquation}, scope::TransferScope, macroname::String, index)
    lhs_scope = uncached(scope)
    map(equations) do eq
        @capture(eq.lhs, name_Symbol[idx_]) || error("$macroname: invalid LHS in `@∑` equation: $(eq.lhs)")
        idx == index || error("$macroname: invalid LHS index in `@∑` equation: $(eq.lhs) (must be [$index])")
        TransferEquation(eq.kind, resolve_refs(eq.lhs, lhs_scope), resolve_refs(eq.rhs, scope), eq.op)
    end
end

function push_unique!(xs::Vector, x)
    x in xs || push!(xs, x)
    xs
end

# `@G2P2G` fuses a G2P and a P2G loop into a single pass over the particles, so
# the support window must be bound once and shared by both halves. The symbol
# travels with a `load` flag: whichever half holds `load=true` emits the
# binding statement, the other just uses it.
struct SupportWindowBinding
    window::Symbol
    load::Bool
end
SupportWindowBinding() = SupportWindowBinding(gensym(:window), true)
SupportWindowBinding(binding::SupportWindowBinding; load::Bool) = SupportWindowBinding(binding.window, load)

function support_window_exprs(binding::SupportWindowBinding, weights, particles, p, grid)
    binding.load || return ()
    (:($(binding.window) = Tesserae.transfer_support_window($weights, $particles, $p, Tesserae.get_mesh($grid))),)
end

# The per-particle support window, straight from the weights storage. A plain
# array of `BasisWeight`s -- the macros accept any container whose `weights[p]`
# is a `BasisWeight` -- goes through the element instead. `BasisWeightArray` is
# itself such an array, so its methods must be the more specific ones.
@inline function transfer_support_window(weights::BasisWeightArray, particles, p, mesh)
    @_propagate_inbounds_meta
    _transfer_support_window(Val(is_lazy(weights)), weights, particles, p, mesh)
end
@inline function transfer_support_window(weights::AbstractArray{<: BasisWeight}, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes(weights[p])
end
@inline function _transfer_support_window(::Val{false}, weights, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes_storage(weights)[p]
end
# Deferred weights store nothing, not even which nodes they cover, so the window
# is derived from the particle position the same way `update!` would have.
@inline function _transfer_support_window(::Val{true}, weights, particles, p, mesh)
    @_propagate_inbounds_meta
    supportnodes(basis(weights), LazyRow(particles, p), mesh)
end

# One weight property's column for particle `p`: the whole leading (support)
# block, with the particle indices fixed. `p` may be a linear index into a
# multi-dimensional particle space (FEM weights are quadrature point x cell),
# so the support-window storage supplies the particle dimensionality.
@inline function weight_prop_view(weights::BasisWeightArray, ::Val{name}, p) where {name}
    @_propagate_inbounds_meta
    storage = supportnodes_storage(weights)
    A = getproperty(weights, name)
    view(A, nfill(:, Val(ndims(A) - ndims(storage)))..., particle_cartesian(storage, p))
end
@inline function weight_prop_view(weights::AbstractArray{<: BasisWeight}, ::Val{name}, p) where {name}
    @_propagate_inbounds_meta
    getproperty(weights[p], name)
end
@inline particle_cartesian(storage, p::Integer) = (@_propagate_inbounds_meta; CartesianIndices(storage)[p])
@inline particle_cartesian(storage, p::CartesianIndex) = p

# Weight references resolve in three steps so one basis evaluation serves every
# referenced property. `weight_columns` is bound once per particle,
# `weight_node_values` once per support node, and each reference reads a field
# out of the result. A stored property gives a view and an index; the deferred
# ones share a single jet, evaluated to the highest order the equations
# reference -- not the order the weights were declared with, which is what makes
# a transfer that only uses `w[ip]` cheap even on `Order(2)` weights.
struct DeferredWeights{K, B, P, M, W}
    basis::B
    pt::P      # the particle row: a basis may need more than the position
    mesh::M
    window::W
end
DeferredWeights{K}(basis::B, pt::P, mesh::M, window::W) where {K, B, P, M, W} =
    DeferredWeights{K, B, P, M, W}(basis, pt, mesh, window)

@inline function deferred_jet(d::DeferredWeights{K}, ip) where {K}
    @_propagate_inbounds_meta
    nodal_basis_jet(Order(K), d.basis, d.pt, d.mesh, d.window[ip])
end

# Derivative order of a deferred property, `nothing` when it is stored, and
# `missing` when the weights have no such property. A plain function on types so
# the generators below can call it -- a generated function may not call another.
function deferred_order(W::Type, name::Symbol)
    W <: BasisWeightArray || return nothing
    Vals = W.parameters[2]
    njets = W.parameters[6].parameters[1] + 1
    pos = findfirst(==(name), fieldnames(Vals))
    pos === nothing && return missing
    pos <= njets && fieldtype(Vals, pos) <: LazyBasisValues ? pos - 1 : nothing
end

function split_weight_properties(W::Type, names)
    stored = Symbol[]; deferred = Symbol[]; orders = Int[]
    for n in names
        k = deferred_order(W, n)
        k === missing && return nothing
        k === nothing ? push!(stored, n) : (push!(deferred, n); push!(orders, k))
    end
    Tuple(stored), Tuple(deferred), orders
end

@generated function weight_columns(weights, ::Val{names}, particles, p, mesh, window) where {names}
    plan = split_weight_properties(weights, names)
    plan === nothing && return :(error("no such weight property among ", $(QuoteNode(names))))
    stored, _, orders = plan
    views = [:(weight_prop_view(weights, Val($(QuoteNode(n))), p)) for n in stored]
    deferred = isempty(orders) ? :nothing :
        :(DeferredWeights{$(maximum(orders))}(basis(weights), LazyRow(particles, p), mesh, window))
    quote
        @_propagate_inbounds_meta
        (stored = NamedTuple{$stored}(($(views...),)), deferred = $deferred)
    end
end

@generated function weight_node_values(weights, cols, ::Val{names}, ip) where {names}
    plan = split_weight_properties(weights, names)
    plan === nothing && return :(error("no such weight property among ", $(QuoteNode(names))))
    _, _, orders = plan
    exprs = map(names) do n
        k = deferred_order(weights, n)
        k === nothing ? :(cols.stored.$n[ip]) : :(jet[$(k + 1)])
    end
    jetexpr = isempty(orders) ? :() : :(jet = deferred_jet(cols.deferred, ip))
    quote
        @_propagate_inbounds_meta
        $jetexpr
        NamedTuple{$names}(($(exprs...),))
    end
end

# One support node of the window, resolved against the grid: an `SpIndex`
# carrying the storage slot for an `SpGrid`, the plain index otherwise.
@inline function transfer_nodeindex(grid::SpGrid, window, ip)
    @_propagate_inbounds_meta
    spinds = get_spinds(grid)
    i = window[ip]
    @boundscheck checkbounds(spinds, i)
    @inbounds spinds[i]
end
@inline function transfer_nodeindex(grid, window, ip)
    @_propagate_inbounds_meta
    window[ip]
end

# How far a block's shared-memory tile reaches past the block, in nodes. A
# particle sits in the block its cell belongs to, and its support window starts
# at most one node before that cell and spans `support_width` nodes, so the
# window can reach one node past the block even for a width-1 basis --
# `BSpline(Constant())` picks the nearest node, which for a particle in the
# block's last cell is the first node of the next block. `p2g_tile_contains`
# states the invariant the block-scheduled kernel relies on; test/transfer.jl
# checks it for every basis.
@inline p2g_tile_halo(support_width::Integer) = max(support_width - 1, 1)

function p2g_tile_contains(basis, mesh::CartesianMesh{dim}, window::CartesianIndices{dim}, block::CartesianIndex{dim}) where {dim}
    bw = blockwidth(mesh)
    halo = p2g_tile_halo(support_width(basis))
    all(ntuple(Val(dim)) do d
        lo = (block[d] - 1) * bw - halo + 1
        lo <= first(window.indices[d]) && last(window.indices[d]) <= lo + bw + 2*halo - 1
    end)
end

# The two lowerings of one @P2G scatter: `particle` writes straight to the
# grid (particle-parallel paths), `tile` accumulates into a shared-memory tile
# whose per-field layout the block-scheduled GPU kernel owns. `names` are the
# scattered grid properties, in equation order.
struct P2GBodies{names, F, FT}
    particle::F
    tile::FT
end
P2GBodies(particle::F, tile::FT, ::Val{names}) where {names, F, FT} = P2GBodies{names, F, FT}(particle, tile)
scattered_names(::P2GBodies{names}) where {names} = names

# Scalar components of one grid value, matching the layout `HybridArray`'s
# flatten/add! use.
@inline tile_components(v::Number) = (v,)
@inline tile_components(v::Union{Tensor, StaticArray}) = Tuple(v)
# Structural, so the generators below can call it: a generator must not call
# other generated functions, which rules out anything touching `zero(T)` for
# Tensorial types.
@inline tile_ncomps(::Type{T}) where {T <: Number} = 1
@inline tile_ncomps(::Type{Tensor{S, T, N, L}}) where {S, T, N, L} = L
@inline tile_ncomps(::Type{SV}) where {SV <: StaticArray} = prod(size(SV))

# A tile stores `SIDE^dim` nodes per scalar component, component-major by
# field. `origin` sits one node before the tile's first node in every axis.
@inline function tile_slot(node::CartesianIndex{dim}, origin::CartesianIndex{dim}, ::Val{SIDE}) where {dim, SIDE}
    slot = node[dim] - origin[dim]
    for d in dim-1:-1:1
        slot = (slot - 1) * SIDE + (node[d] - origin[d])
    end
    slot
end
@inline function tile_node(k::Integer, origin::CartesianIndex{dim}, ::Val{SIDE}) where {dim, SIDE}
    local_index = CartesianIndices(nfill(SIDE, Val(dim)))[k]
    CartesianIndex(ntuple(d -> origin[d] + local_index[d], Val(dim)))
end

@inline function tile_add!(tile, ::Val{TILELEN}, fieldoffset::Int, slot::Integer, v) where {TILELEN}
    data = tile_components(v)
    for j in eachindex(data)
        @inbounds Atomix.@atomic tile[(fieldoffset + j - 1) * TILELEN + slot] += data[j]
    end
end

# The tuple is spelled out because a generated body may not contain closures,
# which rules out `ntuple` with a lambda.
@generated function tile_load(tile, ::Val{TILELEN}, fieldoffset::Int, slot::Integer, ::Type{T}) where {TILELEN, T}
    elems = [:(tile[(fieldoffset + $(j - 1)) * TILELEN + slot]) for j in 1:tile_ncomps(T)]
    :(@inbounds ($(elems...),))
end
@inline tile_rebuild(::Type{T}, comps) where {T <: Number} = comps[1]
@inline tile_rebuild(::Type{T}, comps) where {T <: Union{Tensor, StaticArray}} = T(comps...)

# Merge one tile node into the grid, fully unrolled over the scattered fields
# at compile time: the GPU compiler must see every field eltype as a constant,
# and recursion over the name tuple is too fragile for that.
@generated function merge_tile_node!(grid, tile, ::Val{names}, ::Val{TILELEN}, slot, node) where {names, TILELEN}
    exprs = Any[]
    offset = 0
    for name in names
        T = fieldtype(eltype(grid), name)
        push!(exprs, quote
            comps = tile_load(tile, Val(TILELEN), $offset, slot, $T)
            if !all(iszero, comps)
                @inbounds add!(grid.$name, p2g_write_index(grid, node), tile_rebuild($T, comps))
            end
        end)
        offset += tile_ncomps(T)
    end
    quote
        @_inline_meta
        $(exprs...)
        nothing
    end
end

# Total scalar components scattered per node, and the tile's scalar type. One
# tile holds every scattered field, so their scalar types must agree.
@generated tile_total_comps(grid, ::Val{names}) where {names} =
    sum(name -> tile_ncomps(fieldtype(eltype(grid), name)), names)
@generated function tile_scalartype(grid, ::Val{names}) where {names}
    types = map(name -> eltype(fieldtype(eltype(grid), name)), names)
    allequal(types) || return :(error("@P2G: block-scheduled transfer requires one scalar type across the scattered fields, got ", $(join(unique(types), ", "))))
    first(types)
end

# Grid properties referenced with the node index, straight from the equation
# syntax. The kernels are handed a grid narrowed to these, because every extra
# `SpArray` field in the argument -- referenced or not -- measurably slows a
# GPU transfer kernel down.
function collect_transfer_refs(equations, index)
    names = Symbol[]
    for eq in equations
        for expr in (eq.lhs, eq.rhs)
            MacroTools.postwalk(expr) do ex
                @capture(ex, x_Symbol[i_]) && i == index && push_unique!(names, x)
                ex
            end
        end
    end
    names
end

narrowed_grid_expr(grid, equations, index) =
    :(Tesserae.narrow_transfer_grid($grid, Val($(Expr(:tuple, map(QuoteNode, collect_transfer_refs(equations, index))...)))))

# Keep the mesh (always the first property) so the result stays a grid, and at
# least one array component so an `SpGrid` stays an `SpGrid` for dispatch and
# `get_spinds`. Non-StructArray grids pass through untouched.
narrow_transfer_grid(grid, ::Val) = grid
@generated function narrow_transfer_grid(grid::StructArray{<: Any, <: Any, <: NamedTuple{names}}, ::Val{refs}) where {names, refs}
    keep = Symbol[first(names)]
    for name in Base.tail(names)
        name in refs && push!(keep, name)
    end
    length(keep) == 1 && length(names) > 1 && push!(keep, names[2])
    Tuple(keep) == names && return :grid
    fields = [:(getproperty(grid, $(QuoteNode(name)))) for name in keep]
    :(StructArray(NamedTuple{$(Tuple(keep))}(($(fields...),))))
end

# All four transfer macros take the same shape: an optional `:schedule`
# QuoteNode, three `parent => index` pairs, an optional partition, and the
# equation block. Parsing it once keeps them from drifting, and lets a wrong
# call say what the right one looks like instead of listing `::Any` candidates.
function parse_transfer_macro_args(macroname, args, allow_partition::Bool)
    args = collect(args)
    schedule = QuoteNode(:nothing)
    if !isempty(args) && first(args) isa QuoteNode
        schedule = popfirst!(args)
    end
    partition = nothing
    if length(args) == 5
        allow_partition || throw(ArgumentError(transfer_macro_usage(macroname, allow_partition)))
        partition = args[4]
    elseif length(args) != 4
        throw(ArgumentError(transfer_macro_usage(macroname, allow_partition)))
    end
    schedule, args[1], args[2], args[3], partition, last(args)
end

function transfer_macro_usage(macroname, allow_partition)
    indices = macroname == "@P2G_Matrix" ? "grid=>(i,j) particles=>p weights=>(ip,jp)" :
                                           "grid=>i particles=>p weights=>ip"
    part = allow_partition ? " [partition]" : ""
    "$macroname: expected `$macroname [:schedule] $indices$part begin ... end`"
end

"""
    @P2G grid=>i particles=>p weights=>ip [partition] begin
        equations...
    end

Particle-to-grid transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@P2G grid=>i particles=>p weights=>ip begin

    # Particle-to-grid transfer
    m[i]  = @∑ w[ip] * m[p]
    mv[i] = @∑ w[ip] * m[p] * v[p]
    f[i]  = @∑ -V[p] * σ[p] * ∇w[ip]

    # Calculation on grid
    vⁿ[i] = mv[i] / m[i]
    v[i]  = vⁿ[i] + (f[i] / m[i]) * Δt

end
```

This expands to roughly the following code:

```julia
# Reset grid properties
@. grid.m  = zero(grid.m)
@. grid.mv = zero(grid.mv)
@. grid.f  = zero(grid.f)

# Particle-to-grid transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        grid.m [i] += bw.w[ip] * particles.m[p]
        grid.mv[i] += bw.w[ip] * particles.m[p] * particles.v[p]
        grid.mv[i] += -particles.V[p] * particles.σ[p] * bw.∇w[ip]
    end
end

# Calculation on grid
for i in eachindex(grid)
    grid.vⁿ[i] = grid.mv[i] / grid.m[i]
    grid.v[i]  = grid.vⁿ[i] + (grid.f[i] / grid.m[i]) * Δt
end
```

Use `\$(expr)` inside transfer equations to evaluate an outer expression once
before the generated transfer loops and use the captured value in the loop body.
For example, `\$Δt` captures the current value of `Δt`.

!!! warning
    In `@P2G`, `Calculation on grid` part must be placed after
    `Particle-to-grid transfer` part.
"""
macro P2G(args...)
    P2G_expr(parse_transfer_macro_args("@P2G", args, true)...)
end

function P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, partition, equations::Expr)
    P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), partition, parse_transfer_program(equations))
end

function P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), partition, program::TransferProgram)
    sum_equations, nosum_equations = split_sum_equations(program, "@P2G")

    code = quote
        Tesserae.check_transfer_arguments("@P2G", $grid, $particles, $weights, $partition)
    end

    if !isempty(sum_equations)
        pre, body = P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations)
        tile_names, tile_args, tile_body = P2G_tile_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations)
        if !DEBUG
            body = :(@inbounds $body)
            tile_body = :(@inbounds $tile_body)
        end
        names_val = :(Val($(Expr(:tuple, map(QuoteNode, tile_names)...))))
        code = quote
            $code
            $pre
            Tesserae.P2G(Tesserae.P2GBodies(($grid, $particles, $weights, $p) -> $body,
                                            ($(tile_args...), $grid, $particles, $weights, $p) -> $tile_body,
                                            $names_val),
                         Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, sum_equations, i)), $particles, $weights, $partition)
        end
    end

    if !isempty(nosum_equations)
        body = P2G_nosum_expr((grid,i), nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), $(narrowed_grid_expr(grid, nosum_equations, i)))
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

function P2G_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, binding::SupportWindowBinding=SupportWindowBinding())
    @gensym gridwriteindex
    (; window) = binding

    weight_names = collect_transfer_refs(sum_equations, ip)
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p, particles, grid, window, weight_names)=>ip]; cache=true)
    sum_equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    particle_replacements = cached_replacements(scope, p)
    inner_replacements = cached_replacements(scope, i, ip)
    inner_symbols = p2g_cached_symbols(inner_replacements)

    fillzeros = Any[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in sum_equations
        (; lhs, rhs, op) = eq
        op == :(=)  && push_unique!(fillzeros, :(Tesserae.fillzero!($(remove_indexing(lhs)))))
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        push!(sum_exprs, p2g_sum_add_expr(lhs, gridwriteindex, rhs))
    end

    body = quote
        $(support_window_exprs(binding, weights, particles, p, grid)...)
        $(particle_replacements...)
        $(hoist_exprs...)
        for $ip in eachindex($window)
            $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
            $gridwriteindex = Tesserae.p2g_write_index($grid, $i)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    Expr(:block, fillzeros...), body
end

# The tile lowering of the same scatter equations: writes go through
# `tile_add!` into the workgroup's shared tile at the node's local slot, while
# every read (particles, weights, grid values on the RHS) stays as in the
# particle-parallel body.
function P2G_tile_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector)
    @gensym tile origin sideval tilelenval tileslot
    binding = SupportWindowBinding()
    (; window) = binding

    weight_names = collect_transfer_refs(sum_equations, ip)
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p, particles, grid, window, weight_names)=>ip]; cache=true)
    sum_equations = resolve_sum_equations(sum_equations, scope, "@P2G", i)
    particle_replacements = cached_replacements(scope, p)
    inner_replacements = cached_replacements(scope, i, ip)
    inner_symbols = p2g_cached_symbols(inner_replacements)

    fieldnames = Symbol[]
    offset_syms = Symbol[]
    offset_exprs = Any[]
    hoist_exprs = Any[]
    sum_exprs = Any[]
    for eq in sum_equations
        (; lhs, rhs, op) = eq
        name = tile_field_name(lhs)
        if !(name in fieldnames)
            push!(fieldnames, name)
            sym = Symbol(:tileoffset_, name)
            push!(offset_syms, sym)
            prev = length(offset_exprs) == 0 ? 0 : :($(offset_syms[end-1]) + Tesserae.tile_ncomps(eltype($grid.$(fieldnames[end-1]))))
            push!(offset_exprs, :($sym = $prev))
        end
        op == :(-=) && (rhs = :(-$rhs))
        rhs = hoist_p2g_rhs!(hoist_exprs, inner_symbols, rhs)
        sym = offset_syms[findfirst(==(name), fieldnames)]
        push!(sum_exprs, :(Tesserae.tile_add!($tile, $tilelenval, $sym, $tileslot, $rhs)))
    end

    # The node index is only needed when an equation mentions `i` -- as a grid
    # reference or bare; the tile writes address the shared tile through the
    # local slot alone.
    uses_i = !isempty(scope.replacements[i]) || any(ex -> expr_contains_symbol(ex, i), sum_exprs)
    node_exprs = uses_i ? (:($i = Tesserae.transfer_nodeindex($grid, $window, $ip)),) : ()
    body = quote
        $(offset_exprs...)
        $(support_window_exprs(binding, weights, particles, p, grid)...)
        $(particle_replacements...)
        $(hoist_exprs...)
        for $ip in eachindex($window)
            $(node_exprs...)
            $tileslot = Tesserae.tile_slot($window[$ip], $origin, $sideval)
            $(inner_replacements...)
            $(sum_exprs...)
        end
    end

    args = (tile, origin, sideval, tilelenval)
    fieldnames, args, body
end

function expr_contains_symbol(ex, sym::Symbol)
    found = false
    MacroTools.postwalk(ex) do x
        x === sym && (found = true)
        x
    end
    found
end

function tile_field_name(lhs)
    @capture(lhs, x_.name_[i_]) && return name
    @capture(lhs, x_[i_]) && return x isa Symbol ? x : error("@P2G: cannot derive the scattered field from `$(lhs)`")
    error("@P2G: cannot derive the scattered field from `$(lhs)`")
end

function p2g_cached_symbols(replacements)
    symbols = Set{Symbol}()
    for ex in replacements
        Meta.isexpr(ex, :(=), 2) && ex.args[1] isa Symbol && push!(symbols, ex.args[1])
    end
    symbols
end

function p2g_has_symbol(expr, symbols=nothing)
    if expr isa Symbol
        return symbols === nothing || expr in symbols
    elseif expr isa Expr
        return any(arg -> p2g_has_symbol(arg, symbols), expr.args)
    else
        return false
    end
end

function p2g_simple_factor(expr)
    expr isa Union{Symbol, Number, QuoteNode} && return true
    if Meta.isexpr(expr, :call, 2) && expr.args[1] in (:-, :+)
        return p2g_simple_factor(expr.args[2])
    end
    false
end

function p2g_product_expr(args::Vector)
    length(args) == 1 && return only(args)
    Expr(:call, :*, args...)
end

function hoist_p2g_rhs!(hoist_exprs::Vector, inner_symbols::Set{Symbol}, expr)
    MacroTools.postwalk(expr) do ex
        if Meta.isexpr(ex, :call) && first(ex.args) === :*
            return hoist_p2g_product_runs!(hoist_exprs, inner_symbols, Any[ex.args[2:end]...])
        end
        ex
    end
end

function hoist_p2g_product_runs!(hoist_exprs::Vector, inner_symbols::Set{Symbol}, args::Vector)
    newargs = Any[]
    run = Any[]
    for arg in args
        if p2g_simple_factor(arg) && !p2g_has_symbol(arg, inner_symbols)
            push!(run, arg)
        else
            append_p2g_hoisted_run!(newargs, hoist_exprs, run)
            empty!(run)
            push!(newargs, arg)
        end
    end
    append_p2g_hoisted_run!(newargs, hoist_exprs, run)
    p2g_product_expr(newargs)
end

function append_p2g_hoisted_run!(newargs::Vector, hoist_exprs::Vector, args)
    if length(args) > 1 && all(p2g_simple_factor, args) && any(p2g_has_symbol, args)
        sym = gensym(:p2g_rhs)
        rhs = p2g_product_expr(Any[args...])
        push!(hoist_exprs, :($sym = $rhs))
        push!(newargs, sym)
    else
        append!(newargs, args)
    end
    newargs
end

function p2g_sum_add_expr(lhs, index, rhs)
    Meta.isexpr(lhs, :ref, 2) || error("@P2G: invalid resolved LHS in `@∑` equation: $lhs")
    array, _ = lhs.args
    :(Tesserae.add!($array, $index, $rhs))
end

struct P2GSpGridStorageIndex
    i::Int
end

Base.@propagate_inbounds p2g_write_index(grid, i) = i
Base.@propagate_inbounds p2g_write_index(grid::Grid, i::CartesianIndex) = LinearIndices(grid)[i]
Base.@propagate_inbounds p2g_write_index(grid::SpGrid, i::CartesianIndex) = p2g_write_index(grid, get_spinds(grid)[Tuple(i)...])
@inline function p2g_write_index(::SpGrid, i::SpIndex)
    si = storageindex(i)
    @boundscheck iszero(si) && error("@P2G: inactive SpGrid support node. Call update_sparsity! before @P2G.")
    P2GSpGridStorageIndex(si)
end

@inline function add!(A::SpArray{T}, i::P2GSpGridStorageIndex, v::T) where {T}
    @_propagate_inbounds_meta
    @debug checkbounds(get_data(A), i.i)
    @inbounds get_data(A)[i.i] += v
    A
end

@inline _atomic_index(A::HybridArray{<:Any, <:Any, <:SpArray}, i::P2GSpGridStorageIndex) = i.i

# CPU: sequential
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "@P2G: `ThreadPartition` must be given for threaded computation" maxlog=1

    for p in eachindex(particles)
        @inline f(grid, particles, weights, p)
    end
end

# CPU: multi-threading
function P2G(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ThreadPartition) where {scheduler}
    for group in threadsafe_groups(partition)
        tforeach(group, scheduler) do region
            for p in particle_indices(partition, particles, region)
                @inline f(grid, particles, weights, p)
            end
        end
    end
end

# GPU
# `@Const` is deliberately absent from the container arguments below. It marks
# an argument read-only so the backend can use its read-only cache path, and on
# a plain device array it is free -- it stays on those elsewhere, where the same
# A/B measures it neutral. Applied to a container it is not: KernelAbstractions
# rewrites the argument's type, and indexing a `StructArray`, `BasisWeightArray`,
# or mesh whose arrays have become `Const` wrappers loses far more than the cache
# hint wins. On an RTX 5090 with 614k particles it made `@G2P` 3.4x slower and
# `@P2G` 1.9x, for bit-identical results.
#
# Passing the arrays unpacked and annotating each one is not worth it either:
# the same `@G2P` body over unpacked arrays runs at 0.0648 ms with `@Const` and
# 0.0655 without, against 0.0677 for the container form here. There is nothing
# for the read-only cache to win, because a particle's weights are read by that
# particle alone and the grid reads are already absorbed by L2 -- removing them
# from the kernel entirely does not change its time.
@kernel function gpukernel_P2G(f, grid, particles, weights)
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
end

# Bodies fall back to the particle-parallel lowering everywhere except the
# block-scheduled GPU path below. One method per particle-parallel signature,
# so each is strictly more specific than its `f`-taking sibling.
P2G(bodies::P2GBodies, device::CPUDevice, schedule::Val, grid, particles, weights, ::Nothing) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, nothing)
P2G(bodies::P2GBodies, device::CPUDevice, schedule::Val, grid, particles, weights, partition::ThreadPartition) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, partition)
P2G(bodies::P2GBodies, device::GPUDevice, schedule::Val, grid, particles, weights, ::Nothing) =
    P2G(bodies.particle, device, schedule, grid, particles, weights, nothing)

# GPU with a device partition: one workgroup per nonempty grid block. The
# workgroup zeroes a shared tile, its threads accumulate their block's
# particles into it, and the merged tile is added to the grid with a handful of
# global atomics per node instead of one per particle-node pair. Within one
# block the accumulation order follows the partition's particle order, so the
# result is not bitwise reproducible between runs, exactly like the atomic
# particle-parallel path.
const P2G_BLOCK_GROUPSIZE = 128

@kernel function gpukernel_P2G_blocks(tilebody, grid, particles, weights,
                                      @Const(particleindices), @Const(offsets), @Const(blocklist),
                                      ::Type{Tt}, ::Val{names}, ::Val{SIDE}, ::Val{TILELEN}, ::Val{TOTAL},
                                      ::Val{BW}, ::Val{HALO}, blkdims::Dims{dim}) where {Tt, names, SIDE, TILELEN, TOTAL, BW, HALO, dim}
    grp = @index(Group, Linear)
    l = Int(@index(Local, Linear))
    tile = @localmem Tt (TOTAL,)
    k = l
    while k <= TOTAL
        @inbounds tile[k] = zero(Tt)
        k += P2G_BLOCK_GROUPSIZE
    end
    @synchronize
    @inbounds b = Int(blocklist[grp])
    blockcoord = CartesianIndices(blkdims)[b]
    origin = CartesianIndex(ntuple(d -> (blockcoord[d] - 1) * BW - HALO, Val(dim)))
    @inbounds pstart = Int(offsets[b])
    @inbounds pstop = Int(offsets[b+1])
    k = l
    while k <= pstop - pstart
        @inbounds p = Int(particleindices[pstart + k])
        @inline tilebody(tile, origin, Val(SIDE), Val(TILELEN), grid, particles, weights, p)
        k += P2G_BLOCK_GROUPSIZE
    end
    @synchronize
    gridsize = size(grid)
    k = l
    while k <= TILELEN
        node = tile_node(k, origin, Val(SIDE))
        if all(ntuple(d -> 1 <= node[d] <= gridsize[d], Val(dim)))
            @inline merge_tile_node!(grid, tile, Val(names), Val(TILELEN), k, node)
        end
        k += P2G_BLOCK_GROUPSIZE
    end
end

function P2G(bodies::P2GBodies, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, partition::ThreadPartition{<: GPUBlockStrategy}) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    bs = strategy(partition)
    names = Val(scattered_names(bodies))
    length(bs.particleindices) == length(particles) ||
        error("@P2G: `update!(partition, particles.x)` must run with these particles before the transfer")
    backend = get_backend(device)
    Tt = tile_scalartype(grid, names)
    sw = support_width(basis(weights))
    BW = blockwidth(bs)
    dim = length(nblocks(bs))
    halo = p2g_tile_halo(sw)
    side = BW + 2*halo
    tilelen = side^dim
    total = tilelen * tile_total_comps(grid, names)
    if total * sizeof(Tt) > 32768
        @warn "@P2G: shared-memory tile ($(total * sizeof(Tt)) B) exceeds the block-scheduled budget; falling back to the particle-parallel path" maxlog=1
        return P2G(bodies.particle, device, Val(scheduler), grid, particles, weights, nothing)
    end
    kernel = gpukernel_P2G_blocks(backend, (P2G_BLOCK_GROUPSIZE,))
    kernel(bodies.tile, hybrid(grid), particles, weights,
           bs.particleindices, bs.offsets, bs.blocklist,
           Tt, names, Val(side), Val(tilelen), Val(total), Val(BW), Val(halo), nblocks(bs);
           ndrange=P2G_BLOCK_GROUPSIZE * nactive(bs))
end

G2P2G(f, device::CPUDevice, schedule, grid, particles, weights, partition) =
    P2G(f, device, schedule, grid, particles, weights, partition)

# Unlike P2G, G2P2G writes interpolated and updated particle properties.
@kernel function gpukernel_G2P2G(f, grid, particles, weights)
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function G2P2G(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights, ::Nothing) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_G2P2G(backend)
    kernel(f, hybrid(grid), particles, weights; ndrange=length(particles))
end

function P2G_nosum_expr((grid,i), nosum_equations::Vector)
    scope = TransferScope([grid=>i])
    nosum_equations = map(eq -> resolve_equation(eq, scope), nosum_equations)

    nosum_exprs = map(nosum_equations) do eq
        (; lhs, rhs, op) = eq
        Expr(op, lhs, rhs)
    end
    Expr(:block, nosum_exprs...)
end

# The grid-only part of `@P2G`. It walks the same indices as `@foreach` and
# launches the same GPU kernels, both defined in foreach.jl, but keeps its own
# CPU loops: routing them through `tforeach` costs 5-9% on a dense grid, and
# `@inbounds @simd` here is worth that much.
function P2G_nosum(f, ::CPUDevice, grid)
    @inbounds @simd for i in foreach_indices(grid)
        @inline f(grid, i)
    end
end

function P2G_nosum(f, ::CPUDevice, grid::SpGrid)
    @inbounds for i in foreach_indices(grid)
        @inline f(grid, i)
    end
end

function P2G_nosum(f, device::GPUDevice, grid)
    backend = get_backend(device)
    if grid isa SpGrid
        spinds = get_spinds(grid)
        kernel = gpukernel_foreach_spgrid(backend)
        kernel(f, grid, spinds; ndrange=_spindex_ndrange(spinds))
    else
        kernel = gpukernel_foreach(backend)
        kernel(f, grid; ndrange=size(grid))
    end
end

# `macroname` is threaded through so every transfer macro reports its own name.
# `@G2P` takes no partition and passes `nothing`, which skips the partition
# checks exactly as before.
function check_transfer_arguments(macroname, grid, particles, weights, partition)
    get_mesh(grid) isa AbstractMesh || error("$macroname: grid must have a mesh")
    eltype(weights) <: BasisWeight || error("$macroname: invalid `BasisWeight`s, got type $(typeof(weights))")
    if grid isa SpGrid
        eltype(weights) <: BasisWeight{CPDI} && cpdi_spgrid_error(macroname)
        if length(propertynames(grid)) > 1
            isempty(get_data(getproperty(grid, 2))) && error("$macroname: SpGrid indices not activated")
        end
    end
    @assert length(particles) ≤ length(weights)
    # check device
    device = get_device(grid)
    @assert get_device(particles) == get_device(weights) == device
    check_partition_for_transfer(macroname, device, grid, weights, partition)
end

# A partition must live where the transfer runs: `BlockStrategy` schedules CPU
# threads, `GPUBlockStrategy` schedules GPU workgroups.
check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, ::Nothing) = nothing
check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, ::Nothing) = nothing
function check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, partition::ThreadPartition{<: GPUBlockStrategy})
    macroname == "@P2G" || error("$macroname: the block-scheduled GPU transfer only supports @P2G so far. Use partitionless $macroname on GPU.")
    weights isa BasisWeightArray || error("$macroname: the block-scheduled GPU transfer requires weights from `generate_basis_weights`")
    check_partition_for_transfer(macroname, grid, weights, strategy(partition))
end
function check_partition_for_transfer(macroname, ::GPUDevice, grid, weights, partition)
    error("$macroname: this ThreadPartition lives on the CPU. Transfer it with `gpu(partition)` and `update!` it on the device.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, ::ThreadPartition{<: GPUBlockStrategy})
    error("$macroname: this ThreadPartition lives on the GPU. Construct a CPU one with `ThreadPartition(mesh)`.")
end
function check_partition_for_transfer(macroname, ::CPUDevice, grid, weights, partition::ThreadPartition)
    check_partition_for_transfer(macroname, grid, weights, strategy(partition))
end
check_partition_for_transfer(macroname, grid, weights, strat) = nothing
function check_partition_for_transfer(macroname, grid, weights, strat::BlockStrategy)
    @assert nblocks(get_mesh(grid)) == nblocks(strat)
    if nassigned(strat) == 0
        error("$macroname: No particles assigned to any block in ThreadPartition")
    end
    check_partition_support(macroname, basis(first(weights)), strat)
end
function check_partition_for_transfer(macroname, grid, weights, strat::GPUBlockStrategy)
    @assert nblocks(get_mesh(grid)) == nblocks(strat)
    if nactive(strat) == 0
        error("$macroname: No particles assigned to any block in ThreadPartition")
    end
    check_partition_support(macroname, basis(weights), strat)
end
function check_partition_support(macroname, b, strat)
    if support_width(b) > blockwidth(strat)
        error("$macroname: Block size for `ThreadPartition` is too small for basis $b. Increase `block_size_log2=Val(...)` on the `CartesianMesh` to ensure block size is ≥ kernel support.")
    end
end

"""
    @G2P grid=>i particles=>p weights=>ip begin
        equations...
    end

Grid-to-particle transfer macro.
Based on the `parent => index` expressions, `a[index]` in `equations`
translates to `parent.a[index]`. This `index` can be replaced with
any other name.

# Examples
```julia
@G2P grid=>i particles=>p weights=>ip begin

    # Grid-to-particle transfer
    v[p] += @∑ w[ip] * (vⁿ[i] - v[i])
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]
    x[p] += @∑ w[ip] * v[i] * Δt

    # Calculation on particle
    Δϵₚ = symmetric(∇v[p]) * Δt
    F[p]  = (I + ∇v[p]*Δt) * F[p]
    V[p]  = V⁰[p] * det(F[p])
    σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material

end
```

This expands to roughly the following code:

```julia
# Grid-to-particle transfer
for p in eachindex(particles)
    bw = weights[p]
    nodeindices = supportnodes(bw)
    Δvₚ = zero(eltype(particles.v))
    ∇vₚ = zero(eltype(particles.∇v))
    Δxₚ = zero(eltype(particles.x))
    for ip in eachindex(nodeindices)
        i = nodeindices[ip]
        Δvₚ += bw.w[ip] * (grid.vⁿ[i] - grid.v[i])
        ∇vₚ += grid.v[i] ⊗ bw.∇w[ip]
        Δxₚ += bw.w[ip] * grid.v[i] * Δt
    end
    particles.v[p] += Δvₚ
    particles.∇v[p] = ∇vₚ
    particles.x[p] += Δxₚ
end

# Calculation on particle
for p in eachindex(particles)
    Δϵₚ = symmetric(particles.∇v[p]) * Δt
    particles.F[p]  = (I + particles.∇v[p]*Δt) * particles.F[p]
    particles.V[p]  = particles.V⁰[p] * det(particles.F[p])
    particles.σ[p] += λ*tr(Δϵₚ)*I + 2μ*Δϵₚ # Linear elastic material
end
```

Use `\$(expr)` inside transfer equations to evaluate an outer expression once
before the generated transfer loops and use the captured value in the loop body.
For example, `\$Δt` captures the current value of `Δt`.

!!! warning
    In `@G2P`, `Calculation on particles` part must be placed after
    `Grid-to-particle transfer` part.
"""
macro G2P(args...)
    schedule, grid_i, particles_p, weights_ip, _, equations = parse_transfer_macro_args("@G2P", args, false)
    G2P_expr(schedule, grid_i, particles_p, weights_ip, equations)
end

function G2P_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, equations::Expr)
    G2P_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), parse_transfer_program(equations))
end

function G2P_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), program::TransferProgram)
    sum_equations, nosum_equations = split_sum_equations(program, "@G2P")

    code = quote
        Tesserae.check_transfer_arguments("@G2P", $grid, $particles, $weights, nothing)
    end

    if !isempty(program.equations)
        body = G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations, nosum_equations)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.G2P(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, program.equations, i)), $particles, $weights)
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

# CPU: sequential & multi-threading
function G2P(f, ::CPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    tforeach(eachindex(particles), scheduler) do p
        @inline f(grid, particles, weights, p)
    end
end

# GPU
@kernel function gpukernel_G2P(f, grid, particles, weights)
    p = @index(Global)
    @inline f(grid, particles, weights, p)
end
function G2P(f, device::GPUDevice, ::Val{scheduler}, grid, particles, weights) where {scheduler}
    scheduler == :nothing || @warn "Multi-threading is disabled for GPU" maxlog=1
    particles = particles isa QuadraturePoints ? parent(particles) : particles
    backend = get_backend(device)
    kernel = gpukernel_G2P(backend)
    kernel(f, grid, particles, weights; ndrange=length(particles))
end

function G2P_sum_expr((grid,i), (particles,p), (weights,ip), sum_equations::Vector, nosum_equations::Vector, binding::SupportWindowBinding=SupportWindowBinding())
    (; window) = binding

    code = Expr(:block)
    weight_names = collect_transfer_refs(vcat(sum_equations, nosum_equations), ip)
    scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p, particles, grid, window, weight_names)=>ip]; cache=true)

    if !isempty(sum_equations)
        sum_equations = resolve_sum_equations(sum_equations, scope, "@G2P", p)
        particle_replacements = cached_replacements(scope, p)
        inner_replacements = cached_replacements(scope, i, ip)

        inits = []
        saves = []
        sum_exprs = Any[]
        for eq in sum_equations
            (; lhs, rhs, op) = eq
            tmp = Symbol(lhs, :_p)
            push!(inits, :($tmp = zero(eltype($(remove_indexing(lhs))))))
            push!(saves, Expr(op, lhs, tmp))
            push!(sum_exprs, :($tmp += $rhs))
        end

        code = quote
            $(support_window_exprs(binding, weights, particles, p, grid)...)
            $(particle_replacements...)
            $(inits...)
            for $ip in eachindex($window)
                $i = Tesserae.transfer_nodeindex($grid, $window, $ip)
                $(inner_replacements...)
                $(sum_exprs...)
            end
            $(saves...)
        end
    end

    if !isempty(nosum_equations)
        nosum_scope = TransferScope([grid=>i, particles=>p, TrailingIndexed(weights, p)=>ip])
        nosum_equations = map(eq -> Expr(eq.op, resolve_refs(eq.lhs, nosum_scope), resolve_refs(eq.rhs, nosum_scope)), nosum_equations)
        code = quote
            $code
            $(nosum_equations...)
        end
    end

    code
end

"""
    @G2P2G grid=>i particles=>p weights=>ip [partition] begin
        equations...
    end

Combined grid-to-particle and particle-to-grid transfer macro.

Allows both [`@G2P`](@ref) (interpolation from grid to particles) and [`@P2G`](@ref) (scattering from particles to grid)
to be performed in a single loop over particles, avoiding repeated traversals.

# Examples
```julia
@G2P2G grid=>i particles=>p weights=>ip begin
    # G2P
    ∇v[p] = @∑ v[i] ⊗ ∇w[ip]

    # Particle update
    F[p] = (I + ∇v[p]*Δt) * F[p]
    σ[p] = cauchy_stress(F[p])

    # P2G
    f[i] = @∑ -V[p] * σ[p] * ∇w[ip]
end
```
"""
macro G2P2G(args...)
    G2P2G_expr(parse_transfer_macro_args("@G2P2G", args, true)...)
end

function G2P2G_expr(schedule::QuoteNode, grid_i::Expr, particles_p::Expr, weights_ip::Expr, partition, equations::Expr)
    G2P2G_expr(schedule, unpair(grid_i), unpair(particles_p), unpair(weights_ip), partition, parse_transfer_program(equations))
end

struct G2P2GStages
    g2p_sum::Vector{TransferEquation}
    p2g_sum::Vector{TransferEquation}
    g2p_nosum::Vector{TransferEquation}
    p2g_nosum::Vector{TransferEquation}
end

function split_g2p2g_stages(program::TransferProgram, i, p)
    equations_g2p_sum = TransferEquation[]
    equations_p2g_sum = TransferEquation[]
    equations_g2p_nosum = TransferEquation[]
    equations_p2g_nosum = TransferEquation[]
    precedence = 1
    for eq in program.equations
        if is_sum(eq)
            @capture(eq.lhs, A_[index_]) || error("@G2P2G: invalid LHS in `@∑` equation: $(eq.lhs)")
            if index == p
                precedence == 1 || error("@G2P2G: particle `@∑` equations must come before particle updates and grid-scattering equations")
                push!(equations_g2p_sum, eq)
            elseif index == i
                precedence ≤ 3 || error("@G2P2G: grid `@∑` equations must come before grid-only equations")
                push!(equations_p2g_sum, eq)
                precedence = 3
            else
                error("@G2P2G: wrong index in LHS equation, $(eq.lhs)")
            end
        else
            if precedence in (1, 2)
                push!(equations_g2p_nosum, eq)
                precedence = 2
            elseif precedence in (3, 4)
                push!(equations_p2g_nosum, eq)
                precedence = 4
            else
                error("unreachable")
            end
        end
    end
    G2P2GStages(equations_g2p_sum, equations_p2g_sum, equations_g2p_nosum, equations_p2g_nosum)
end

function G2P2G_expr(schedule::QuoteNode, (grid,i), (particles,p), (weights,ip), partition, program::TransferProgram)
    stages = split_g2p2g_stages(program, i, p)

    code = quote
        Tesserae.check_transfer_arguments("@G2P2G", $grid, $particles, $weights, $partition)
    end
    body = Expr(:block)
    binding = SupportWindowBinding()

    if !isempty(stages.g2p_sum) || !isempty(stages.g2p_nosum)
        expr = G2P_sum_expr((grid,i), (particles,p), (weights,ip), stages.g2p_sum, stages.g2p_nosum, binding)
        body = quote
            $body
            $expr
        end
    end

    if !isempty(stages.p2g_sum)
        # `G2P_sum_expr` binds the basis weight only when it has `@∑` equations,
        # so this half must load it itself otherwise.
        p2g_binding = isempty(stages.g2p_sum) ? binding : SupportWindowBinding(binding; load=false)
        pre, expr = P2G_sum_expr((grid,i), (particles,p), (weights,ip), stages.p2g_sum, p2g_binding)
        code = quote
            $code
            $pre
        end
        body = quote
            $body
            $expr
        end
    end

    if !DEBUG
        body = :(@inbounds $body)
    end
    particle_equations = vcat(collect(stages.g2p_sum), collect(stages.g2p_nosum), collect(stages.p2g_sum))
    code = quote
        $code
        Tesserae.G2P2G(($grid, $particles, $weights, $p) -> $body, Tesserae.get_device($grid), Val($schedule), $(narrowed_grid_expr(grid, particle_equations, i)), $particles, $weights, $partition)
    end

    if !isempty(stages.p2g_nosum)
        body = P2G_nosum_expr((grid,i), stages.p2g_nosum)
        if !DEBUG
            body = :(@inbounds $body)
        end
        code = quote
            $code
            Tesserae.P2G_nosum(($grid, $i) -> $body, Tesserae.get_device($grid), $(narrowed_grid_expr(grid, stages.p2g_nosum, i)))
        end
    end

    code = interpolate_transfer_values(code, program)
    esc(prettify(code; lines=true, alias=false))
end

####################
# Helper functions #
####################

function unpair(ex)
    if @capture(ex, lhs_Symbol => rhs_Symbol)
        return (lhs, rhs)
    elseif @capture(ex, lhs_Symbol => (rhs1_Symbol,rhs2_Symbol))
        return lhs, (rhs1, rhs2)
    elseif @capture(ex, (lhs1_Symbol,lhs2_Symbol) => (rhs1_Symbol,rhs2_Symbol))
        return (lhs1, lhs2), (rhs1, rhs2)
    else
        error("invalid expression, $ex")
    end
end

function has_sum_macro(expr)
    has_sum = Ref(false)
    MacroTools.postwalk(expr) do ex
        if Meta.isexpr(ex, :macrocall, 2) && (ex.args[1]==Symbol("@∑") || ex.args[1]==Symbol("@Σ"))
            has_sum[] = true
        end
        ex
    end
    has_sum[]
end

function parse_transfer_program(expr::Expr)
    expr = MacroTools.prewalk(MacroTools.rmlines, expr)
    @capture(expr, begin exprs__ end) || error("expected a `begin ... end` block, got $expr")
    interpolations = Pair{Symbol, Any}[]
    equations = map(exprs) do ex
        dict = MacroTools.trymatch(Expr(:op_, :lhs_, :rhs_), ex)
        dict === nothing && error("wrong expression: $ex")
        lhs, rhs, op = dict[:lhs], dict[:rhs], dict[:op]
        has_transfer_interpolation(lhs) && error("transfer interpolation with `\$` is only allowed on the RHS, got LHS `$lhs`")
        rhs = extract_transfer_interpolations(rhs, interpolations)
        if @capture(rhs, @∑ eq_)
            (op == :(=) || op == :(+=) || op == :(-=)) || error("@∑ is only allowed on the RHS of assignments with `=`, `+=`, or `-=`, got $ex")
            return TransferEquation(:sum, lhs, eq, op)
        end
        has_sum_macro(rhs) && error("@∑ must appear alone as the entire RHS expression, got $ex")
        TransferEquation(:assign, lhs, rhs, op)
    end
    TransferProgram(equations, interpolations)
end

function has_transfer_interpolation(expr)
    Meta.isexpr(expr, :$, 1) && return true
    expr isa Expr || return false
    any(has_transfer_interpolation, expr.args)
end

function extract_transfer_interpolations(expr, interpolations)
    if Meta.isexpr(expr, :$, 1)
        captured = gensym(:transfer_interp)
        push!(interpolations, captured => only(expr.args))
        return captured
    elseif expr isa Expr
        return Expr(expr.head, map(arg -> extract_transfer_interpolations(arg, interpolations), expr.args)...)
    else
        return expr
    end
end

function interpolate_transfer_values(code, program::TransferProgram)
    isempty(program.interpolations) && return code
    bindings = map(program.interpolations) do captured_rhs
        captured, rhs = captured_rhs
        Expr(:(=), captured, rhs)
    end
    Expr(:let, Expr(:block, bindings...), code)
end

function resolve_refs(expr, scope::TransferScope)
    MacroTools.postwalk(expr) do ex
        if @capture(ex, x_[i_]) && haskey(scope.bindings, i)
            parent = scope.bindings[i]
            if parent isa TrailingIndexed
                if scope.replacements === nothing
                    return :($(parent.parent).$x[$i, $(parent.trailing)])
                end
                # One binding per particle and one per node, both identical across
                # every weight reference, so `push_unique!` emits each once and
                # the properties share a single basis evaluation.
                push_unique!(scope.replacements[parent.trailing],
                             :($(parent.cols) = Tesserae.weight_columns($(parent.parent), Val($(parent.names)), $(parent.particles), $(parent.trailing), Tesserae.get_mesh($(parent.grid)), $(parent.window))))
                push_unique!(scope.replacements[i],
                             :($(parent.vals) = Tesserae.weight_node_values($(parent.parent), $(parent.cols), Val($(parent.names)), $i)))
                return :($(parent.vals).$x)
            else
                resolved = :($parent.$x[$i])
                sym = Symbol(resolved)
            end
            scope.replacements === nothing && return resolved
            push_unique!(scope.replacements[i], :($sym = $resolved))
            return sym
        end
        ex
    end
end

function remove_indexing(expr)
    MacroTools.postwalk(expr) do ex
        @capture(ex, x_[i__]) && return x
        ex
    end
end

