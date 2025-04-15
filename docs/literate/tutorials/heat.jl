# # Heat problem using FEM
#
# ```@raw html
# <img src="https://github.com/user-attachments/assets/74a9fad8-da97-4ce6-8217-838fe96660ac" width="400"/>
# ```

using Tesserae

function main()

    ## Properties for grid and particles
    GridProp = @NamedTuple begin
        x :: Vec{2, Float64} # Coordinate
        u :: Float64         # Temperature
        f :: Float64         # Heat source
    end
    ParticleProp = @NamedTuple begin
        x      :: Vec{2, Float64} # Coordinate
        detJdV :: Float64         # Weighted det(J) for Gauss–Legendre quadrature
    end

    ## FEM mesh using UnstructuredMesh
    mesh = UnstructuredMesh(CartesianMesh(0.1, (-1,1), (-1,1)))
    grid = generate_grid(GridProp, mesh)

    ## Integration points
    particles = generate_particles(ParticleProp, mesh)

    ## Interpolation
    mpvalues = generate_mpvalues(mesh, size(particles); name=Val(:N))
    feupdate!(mpvalues, mesh; volume = particles.detJdV) # Use `feupdate!` instead of `update!`

    ## Global matrix
    ndofs = 1 # Degrees of freedom per node
    K = create_sparse_matrix(mesh; ndofs)

    ## Create dofmap considering boundary conditions
    dofmask = trues(ndofs, size(grid)...)
    dofmask[1, findall(x -> x[1]==-1 || x[1]==1, mesh)] .= false
    dofmask[1, findall(x -> x[2]==-1 || x[2]==1, mesh)] .= false
    dofmap = DofMap(dofmask)

    ## Construct global vector (on grid) and matrix
    @P2G grid=>i particles=>p mpvalues=>ip begin
        f[i] = @∑ N[ip] * detJdV[p]
    end
    @P2G_Matrix grid=>(i,j) particles=>p mpvalues=>(ip,jp) begin
        K[i,j] = @∑ ∇N[ip] ⋅ ∇N[jp] * detJdV[p]
    end

    ## Solve the equation
    dofmap(grid.u) .= extract(K, dofmap) \ Array(dofmap(grid.f))

    ## Output the results
    openvtk("heat", mesh) do vtk
        vtk["Temperature"] = grid.u
    end
    norm(grid.u) #src
end

using Test                            #src
if @isdefined(RUN_TESTS) && RUN_TESTS #src
    @test main() ≈ 3.3077439126413104 #src
end                                   #src
