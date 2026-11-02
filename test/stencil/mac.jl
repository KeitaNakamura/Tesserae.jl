function periodic_difference!(out, a, R, d, hd)
    e = unitoffsets(R)[d]
    L, U = boundaries(R, d)
    @stencil begin
        out[R] = (a[R+e/2] - a[R-e/2])/hd
        out[L] = (a[L+e/2] - a[U-e/2])/hd
        out[U] = out[L]
    end
    out
end

function mac_operators(p, u, v, h; periodic)
    C = Region(Cell(), size(p))
    V = Region(Vertex(), C)
    faces = [Region(Face(d), C) for d in 1:2]
    offsets = unitoffsets(C)
    velocity = [u, v]
    gradient = [similar(p, size(F)) for F in faces]
    normal = [similar(p) for _ in velocity]
    transverse = [similar(p, size(V)) for _ in velocity]
    laplacian = [similar(a) for a in velocity]
    divergence = similar(p)
    fill!(divergence, 0)

    for (d, e) in enumerate(offsets)
        j = 3-d
        et, hd, ht = offsets[j], h[d], h[j]
        F = faces[d]
        L, R = boundaries(F)
        a, g, dn, ds, lap = velocity[d], gradient[d], normal[d], transverse[d], laplacian[d]

        if periodic
            @stencil a[R] = a[L]
            periodic_difference!(g, p, F, d, hd)
        else
            @stencil begin
                a[L] = 0
                a[R] = 0
                g[F] = (p[F+e/2] - p[F-e/2])/hd
                g[L] = 0
                g[R] = 0
            end
        end
        @stencil begin
            dn[C] = (a[C+e/2] - a[C-e/2])/hd
            divergence[C] += dn[C]
        end

        if periodic
            periodic_difference!(ds, a, V, j, ht)
            periodic_difference!(lap, dn, F, d, hd)
            @stencil lap[F] += (ds[F+et/2] - ds[F-et/2])/ht
        else
            # Tangential no-slip walls are half a cell from the velocity samples.
            B, T = boundaries(V, j)
            @stencil begin
                ds[V] = (a[V+et/2] - a[V-et/2])/ht
                ds[B] =  a[B+et/2]/(ht/2)
                ds[T] = -a[T-et/2]/(ht/2)
            end
            fill!(lap, 0)
            @stencil lap[F] = (dn[F+e/2] - dn[F-e/2])/hd + (ds[F+et/2] - ds[F-et/2])/ht
        end
    end

    (; gx=gradient[1], gy=gradient[2], divergence,
       du_dx=normal[1], dv_dy=normal[2], du_dy=transverse[1], dv_dx=transverse[2],
       lap_u=laplacian[1], lap_v=laplacian[2])
end

# Reference stencils written with ordinary integer indices.
function reference_operators(p, u0, v0, h; periodic)
    nx,ny = size(p)
    hx,hy = h
    u,v = copy(u0),copy(v0)
    if periodic
        u[end,:] .= u[1,:]
        v[:,end] .= v[:,1]
    else
        u[[1,end],:] .= 0
        v[:,[1,end]] .= 0
    end
    gx,gy = zeros(nx+1,ny),zeros(nx,ny+1)
    for j in 1:ny, i in 1:nx+1
        if periodic
            gx[i,j] = (p[mod1(i,nx),j]-p[mod1(i-1,nx),j])/hx
        elseif 1 < i < nx+1
            gx[i,j] = (p[i,j]-p[i-1,j])/hx
        end
    end
    for j in 1:ny+1, i in 1:nx
        if periodic
            gy[i,j] = (p[i,mod1(j,ny)]-p[i,mod1(j-1,ny)])/hy
        elseif 1 < j < ny+1
            gy[i,j] = (p[i,j]-p[i,j-1])/hy
        end
    end
    divergence = [(u[i+1,j]-u[i,j])/hx+(v[i,j+1]-v[i,j])/hy
                  for i in 1:nx, j in 1:ny]
    du_dy,dv_dx = zeros(nx+1,ny+1),zeros(nx+1,ny+1)
    for j in 1:ny+1, i in 1:nx+1
        if periodic
            du_dy[i,j] = (u[i,mod1(j,ny)]-u[i,mod1(j-1,ny)])/hy
            dv_dx[i,j] = (v[mod1(i,nx),j]-v[mod1(i-1,nx),j])/hx
        else
            if 1 < i < nx+1
                du_dy[i,j] = j == 1 ? 2u[i,1]/hy :
                    j == ny+1 ? -2u[i,ny]/hy : (u[i,j]-u[i,j-1])/hy
            end
            if 1 < j < ny+1
                dv_dx[i,j] = i == 1 ? 2v[1,j]/hx :
                    i == nx+1 ? -2v[nx,j]/hx : (v[i,j]-v[i-1,j])/hx
            end
        end
    end
    lap_u,lap_v = zeros(size(u)),zeros(size(v))
    for j in 1:ny, i in 1:nx+1
        if periodic
            lap_u[i,j] =
                (u[mod1(i+1,nx),j]-2u[i,j]+u[mod1(i-1,nx),j])/hx^2 +
                (u[i,mod1(j+1,ny)]-2u[i,j]+u[i,mod1(j-1,ny)])/hy^2
        elseif 1 < i < nx+1
            below = j == 1 ? -u[i,j] : u[i,j-1]
            above = j == ny ? -u[i,j] : u[i,j+1]
            lap_u[i,j] = (u[i+1,j]-2u[i,j]+u[i-1,j])/hx^2 +
                (above-2u[i,j]+below)/hy^2
        end
    end
    for j in 1:ny+1, i in 1:nx
        if periodic
            lap_v[i,j] =
                (v[mod1(i+1,nx),j]-2v[i,j]+v[mod1(i-1,nx),j])/hx^2 +
                (v[i,mod1(j+1,ny)]-2v[i,j]+v[i,mod1(j-1,ny)])/hy^2
        elseif 1 < j < ny+1
            left = i == 1 ? -v[i,j] : v[i-1,j]
            right = i == nx ? -v[i,j] : v[i+1,j]
            lap_v[i,j] = (right-2v[i,j]+left)/hx^2 +
                (v[i,j+1]-2v[i,j]+v[i,j-1])/hy^2
        end
    end
    (; u,v,gx,gy,divergence,du_dy,dv_dx,lap_u,lap_v)
end

@testset "MAC operators against integer-index stencils" begin
    rng = MersenneTwister(25)
    for n in ((7,5),(2,3),(1,4),(4,1),(1,1)), periodic in (false,true)
        nx,ny = n
        h = (0.13,0.21)
        p,u,v = randn(rng,n),randn(rng,nx+1,ny),randn(rng,nx,ny+1)
        p_before,u_before,v_before = copy(p),copy(u),copy(v)
        expected = reference_operators(p,u,v,h; periodic)
        result = mac_operators(p,u,v,h; periodic)
        @test p == p_before
        @test u == expected.u
        @test v == expected.v
        for field in (:gx,:gy,:divergence,:du_dy,:dv_dx,:lap_u,:lap_v)
            actual,reference = getproperty(result,field),getproperty(expected,field)
            # Allow only rounding differences caused by the order of arithmetic operations.
            scale = max(1,maximum(abs,reference))
            @test maximum(abs,actual-reference) ≤ 64eps(Float64)*scale
            @test all(isfinite,actual)
        end
        @test u[2:nx,:] == u_before[2:nx,:]
        @test v[:,2:ny] == v_before[:,2:ny]

        # Summed divergence equals the boundary flux, zero for both boundary conditions.
        @test abs(sum(result.divergence)) ≤
            64eps(Float64)*max(1,sum(abs,result.divergence))

        # Discrete integration by parts: gradient is the negative adjoint of divergence.
        # Count duplicated periodic faces only once.
        if periodic
            pressure_work = sum(u[1:nx,:].*result.gx[1:nx,:]) +
                sum(v[:,1:ny].*result.gy[:,1:ny])
            viscous_work = sum(u[1:nx,:].*result.lap_u[1:nx,:]) +
                sum(v[:,1:ny].*result.lap_v[:,1:ny])
        else
            pressure_work = sum(u.*result.gx)+sum(v.*result.gy)
            viscous_work = sum(u.*result.lap_u)+sum(v.*result.lap_v)
        end
        cell_work = sum(p.*result.divergence)
        @test abs(pressure_work+cell_work) ≤
            128eps(Float64)*max(1,abs(pressure_work),sum(abs,p.*result.divergence))
        @test viscous_work ≤ 0
    end
end

function manufactured_errors(n; periodic)
    nx,ny = n
    hx,hy = 1/nx,1/ny
    xc,yc = ((1:nx).-0.5).*hx,((1:ny).-0.5).*hy
    xf,yf = (0:nx).*hx,(0:ny).*hy
    if periodic
        p = [sinpi(2x)+0.5sinpi(2y) for x in xc,y in yc]
        u = [cospi(2x)*cospi(2y) for x in xf,y in yc]
        v = [sinpi(2x)*cospi(2y) for x in xc,y in yf]
        r = mac_operators(p,u,v,(hx,hy); periodic=true)
        kx,ky = 2sinpi(hx)/hx,2sinpi(hy)/hy
        gx = [kx*cospi(2x) for x in xf,y in yc]
        gy = [0.5ky*cospi(2y) for x in xc,y in yf]
        divergence = [-kx*sinpi(2x)*cospi(2y)-ky*sinpi(2x)*sinpi(2y)
                      for x in xc,y in yc]
        eigenvalue = (2sinpi(hx)/hx)^2+(2sinpi(hy)/hy)^2
        @test maximum(abs,r.gx-gx) < 2e-12
        @test maximum(abs,r.gy-gy) < 2e-12
        @test maximum(abs,r.divergence-divergence) < 2e-12
        @test maximum(abs,r.lap_u+eigenvalue*u) < 2e-11
        @test maximum(abs,r.lap_v+eigenvalue*v) < 2e-11
        exact_gx = [2π*cospi(2x) for x in xf,y in yc]
        exact_gy = [π*cospi(2y) for x in xc,y in yf]
        exact_div = [-2π*sinpi(2x)*cospi(2y)-2π*sinpi(2x)*sinpi(2y)
                     for x in xc,y in yc]
    else
        p = [cospi(x)*cospi(y) for x in xc,y in yc]
        u = [x*(1-x)*y*(1-y) for x in xf,y in yc]
        v = [2x*(1-x)*y*(1-y) for x in xc,y in yf]
        r = mac_operators(p,u,v,(hx,hy); periodic=false)
        gx = [-2sinpi(hx/2)/hx*sinpi(x)*cospi(y) for x in xf,y in yc]
        gy = [-2sinpi(hy/2)/hy*cospi(x)*sinpi(y) for x in xc,y in yf]
        exact_div = [(1-2x)*y*(1-y)+2x*(1-x)*(1-2y) for x in xc,y in yc]
        @test maximum(abs,r.gx-gx) < 2e-12
        @test maximum(abs,r.gy-gy) < 2e-12
        @test maximum(abs,r.divergence-exact_div) < 2e-12
        exact_gx = [-π*sinpi(x)*cospi(y) for x in xf,y in yc]
        exact_gy = [-π*cospi(x)*sinpi(y) for x in xc,y in yf]
    end
    gradient_error = max(maximum(abs,r.gx-exact_gx),maximum(abs,r.gy-exact_gy))
    divergence_error = maximum(abs,r.divergence-exact_div)
    (; gradient_error,divergence_error)
end

@testset "Analytic fields and spatial refinement" begin
    wall_coarse = manufactured_errors((16,12);periodic=false)
    wall_fine = manufactured_errors((32,24);periodic=false)
    periodic_coarse = manufactured_errors((16,12);periodic=true)
    periodic_fine = manufactured_errors((32,24);periodic=true)
    @test 3.8 < wall_coarse.gradient_error/wall_fine.gradient_error < 4.2
    @test 3.8 < periodic_coarse.gradient_error/periodic_fine.gradient_error < 4.2
    @test 3.8 < periodic_coarse.divergence_error/periodic_fine.divergence_error < 4.2
end

@testset "Periodic constant fields" begin
    p,u,v = fill(2.0,7,5),fill(3.0,8,5),fill(-4.0,7,6)
    r = mac_operators(p,u,v,(0.13,0.21); periodic=true)
    for field in propertynames(r)
        @test all(iszero,getproperty(r,field))
    end
end

@testset "Wall and periodic MAC with a backend defining view(A, I...)" begin
    rng = MersenneTwister(91)
    for n in ((7,5), (1,4), (4,1), (1,1)), periodic in (false, true)
        nx,ny = n
        p = randn(rng, n)
        u,v = randn(rng, nx+1, ny),randn(rng, nx, ny+1)
        h = (0.13,0.21)
        expected = mac_operators(p, copy(u), copy(v), h; periodic)
        pressure = TrackedArray(copy(p), Ref(0))
        horizontal = TrackedArray(copy(u), Ref(0))
        vertical = TrackedArray(copy(v), Ref(0))
        actual = mac_operators(pressure, horizontal, vertical, h; periodic)
        @test pressure.data == p
        @test all(field -> getproperty(actual, field) isa TrackedArray, propertynames(actual))
        for field in propertynames(expected)
            @test getproperty(actual, field).data == getproperty(expected, field)
        end
    end
end
