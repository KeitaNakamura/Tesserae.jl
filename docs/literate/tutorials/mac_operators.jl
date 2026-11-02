# # MAC operators with wall and periodic boundaries
#
# Pressure gradients, divergence, and velocity Laplacians on a staggered grid.
# This example defines the discrete operators; it does not solve a time step.
# Scalars, regions, allocations, and standalone helper calls are prepared outside each
# `@stencil` block. Blocks group array updates, executed in the written order;
# every equation independently selects its valid calculation points.

using Tesserae.Stencil

"""
この使用例で使う周期差分。方向dで中央配置されたaを、格子線上のRへ微分します。
内部差分・周期の継ぎ目・重複面の同期を、通常のviewとbroadcastで記述します。
ライブラリへの追加APIではありません。
"""
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

"""
MAC演算子の使用例。pはセル中心、uはx面、vはy面の通常の配列です。
静止壁でu・vにno-slip、pに法線勾配0を指定します。
u・vの壁上の法線成分を更新します。圧力Poisson方程式は解きません。
"""
function wall_operators(p, u, v, h)
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

    # d=1はuとx方向、d=2はvとy方向。jはもう一方の接線方向です。
    for (d, e) in enumerate(offsets)
        j = 3-d
        et, hd, ht = offsets[j], h[d], h[j]
        F = faces[d]
        L, R = boundaries(F)
        a, g, dn, ds, lap = velocity[d], gradient[d], normal[d], transverse[d], laplacian[d]

        # 壁上の法線速度は0、圧力の法線勾配も0。
        @stencil begin
            a[L] = 0
            a[R] = 0
            g[F] = (p[F+e/2] - p[F-e/2])/hd
            g[L] = 0
            g[R] = 0

            # 発散の各方向の寄与。
            dn[C] = (a[C+e/2] - a[C-e/2])/hd
            divergence[C] += dn[C]
        end

        # 接線速度のno-slipは、壁までの半セル距離で勾配を指定。
        B, T = boundaries(V, j)
        @stencil begin
            ds[V] = (a[V+et/2] - a[V-et/2])/ht
            ds[B] =  a[B+et/2]/(ht/2)
            ds[T] = -a[T-et/2]/(ht/2)
        end

        fill!(lap, 0)
        @stencil lap[F] =
            (dn[F+e/2] - dn[F-e/2])/hd + (ds[F+et/2] - ds[F-et/2])/ht
    end

    (; gx=gradient[1], gy=gradient[2], divergence,
       du_dx=normal[1], dv_dy=normal[2], du_dy=transverse[1], dv_dx=transverse[2],
       lap_u=laplacian[1], lap_v=laplacian[2])
end

"""
同じMAC配置で両方向を周期境界にした使用例。
上下・左右は同じ物理面なので、uの右端とvの上端を対応する下限側の値に同期します。
"""
function periodic_operators(p, u, v, h)
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

        @stencil a[R] = a[L]
        periodic_difference!(g, p, F, d, hd)

        @stencil begin
            dn[C] = (a[C+e/2] - a[C-e/2])/hd
            divergence[C] += dn[C]
        end

        periodic_difference!(ds, a, V, j, ht)
        periodic_difference!(lap, dn, F, d, hd)
        @stencil lap[F] += (ds[F+et/2] - ds[F-et/2])/ht
    end

    (; gx=gradient[1], gy=gradient[2], divergence,
       du_dx=normal[1], dv_dy=normal[2], du_dy=transverse[1], dv_dx=transverse[2],
       lap_u=laplacian[1], lap_v=laplacian[2])
end

function mac_example(n=(32, 24))
    C = Region(Cell(), n)
    Fx, Fy = Region(Face(1), C), Region(Face(2), C)
    h = 1 ./ n

    # 壁境界：圧力は壁で法線勾配0、速度は全ての壁で0となる滑らかな入力。
    p = [cospi(x)*cospi(y) for (x, y) in coordinates(C, h)]
    u = [x*(1-x)*y*(1-y) for (x, y) in coordinates(Fx, h)]
    v = [2*x*(1-x)*y*(1-y) for (x, y) in coordinates(Fy, h)]
    walls = wall_operators(p, u, v, h)

    # 周期境界：両端の値が一致する三角関数の入力。
    pp = [sinpi(2*x)+0.5*sinpi(2*y) for (x, y) in coordinates(C, h)]
    up = [cospi(2*x)*cospi(2*y) for (x, y) in coordinates(Fx, h)]
    vp = [sinpi(2*x)*cospi(2*y) for (x, y) in coordinates(Fy, h)]
    periodic = periodic_operators(pp, up, vp, h)

    # 壁では境界面流量0、周期では反対側の面流量が相殺されます。
    println("cells: ", n)
    println("wall net flux: ", sum(walls.divergence)*prod(h))
    println("periodic net flux: ", sum(periodic.divergence)*prod(h))
    (; walls, periodic)
end

if abspath(PROGRAM_FILE) == @__FILE__
    mac_example()
end
