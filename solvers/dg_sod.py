"""
 * ----------------------------------------------
 * Copyright (c) {2024} Qiming Wang, Xiamen University
 * ----------------------------------------------
 * @file    :   dg_sod.py
 * @date    :   2024/07/31 20:56:49
 * @author  :   Qiming Wang
 * @brief   :   Nodal DG scheme for the BGK equation with macro-micro decomposition on one mesh
 *
"""
import numpy as np
import time
import os

# Constants
nd = 400
mnm = 3
md = 9
ndm = nd + 3
mg = 5
nvm = 99
tol = 1e-6
sft = 3

# initialize
def initdata():
    # Allocate the global variables
    # /solution: u, g /tmp: u0, g0, utem, uc, gc /flux: flxu, flxg, flxmu, flxmg /grid: x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    # /gauss: xq, cq, xp, wt, ai /rk_coefficient: rk_ex, rk_im /parameter: mp, mn, mo, i_bc /integer: kcmax, kcount, nx
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    # /time: t, dt, tfinal, eps, pi /CFL: cflc, cfld, em /vel: vmax /heat_ratio: gamma, gm1 /hybrid: indk
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    u = np.zeros((md + 1, ndm + 4, mnm + 1))
    g = np.zeros((md + 1, ndm + 4, nvm + 1))
    u0 = np.zeros((md + 1, ndm + 4, mnm + 1))
    g0 = np.zeros((md + 1, ndm + 4, nvm + 1))
    utem = np.zeros((md + 1, ndm + 4))
    uc = np.zeros((md + 1, ndm + 4, mnm + 1, 4))
    gc = np.zeros((md + 1, ndm + 4, nvm + 1, 4))
    flxu = np.zeros((md + 1, ndm + 4, mnm + 1, 4))
    flxg = np.zeros((md + 1, ndm + 4, mnm + 1, 4))
    flxmu = np.zeros((md + 1, ndm + 4, nvm + 1, 4))
    flxmg = np.zeros((md + 1, ndm + 4, nvm + 1, 4))
    x = np.zeros(ndm + 4)
    v = np.zeros(nvm+1)
    vq = np.zeros(nvm+1)
    wq = np.zeros(nvm+1)
    xq = np.zeros(mg)
    cq = np.zeros(mg)
    xp = np.zeros(md+1)
    wt = np.zeros(md+1)
    ai = np.zeros(md+1)
    rk_ex = np.zeros((4, 4))
    rk_im = np.zeros((4, 4))
    kcount = 0
    t = 0.0
    dt = 0.0
    tfinal = 0.1
    pi = np.pi
    cfld = 0.0

    # initdata: set up the necessary data before setting the initial condition
    # Gauss quadrature for integration
    xq = [0.906179845938664, 0.538469310105683, 0.0, -0.538469310105683, -0.906179845938664]
    cq = [0.236926885056189, 0.478628670499366, 0.568888888888889, 0.478628670499366, 0.236926885056189]
    rk_ex = np.array([
        [0.5, 0.0, 0.0, 0.0],
        [0.611111111111111, 0.055555555555556, 0.0, 0.0],
        [0.833333333333333, -0.833333333333333, 0.5, 0.0],
        [0.25, 1.75, 0.75, -1.75]
    ])
    rk_im = np.array([
        [0.5, 0.0, 0.0, 0.0],
        [0.16666666666667, 0.5, 0.0, 0.0],
        [-0.5, 0.5, 0.5, 0.0],
        [1.5, -1.5, 0.5, 0.5]
    ])
    [vq.__setitem__(i, -1.+i*2./nvm) for i in range(0, nvm+1)]
    eps = 0.01
    mo = 3
    mp = mo - 1
    mn = mnm
    cflc = 0.05
    tfinal = 0.1
    i_bc = 3
    kcmax = 100000000
    # nodal DG Gauss quadrature points and weights
    if mp==0:
        xp[0] = 0.
        wt[0] = 1.
    elif mp==1:
        xp[0] = -1. / np.sqrt(3.) / 2.
        xp[1] = -xp[0]
        wt[0] = 1. / 2.
        wt[1] = wt[0]
    elif mp==2:
        xp[0] = -np.sqrt(3. / 5.) / 2.
        xp[1] = 0.
        xp[2] = -xp[0]
        wt[0] = 5. / 18.
        wt[1] = 4. / 9.
        wt[2] = wt[0]
    elif mp==3:
        xp[0] = -np.sqrt((3. + 2. * np.sqrt(6. / 5.)) / 7.) / 2.
        xp[1] = -np.sqrt((3. - 2. * np.sqrt(6. / 5.)) / 7.) / 2.
        xp[2] = -xp[1]
        xp[3] = -xp[0]
        wt[0] = (18. - np.sqrt(30.)) / 72.
        wt[1] = (18. + np.sqrt(30.)) / 72.
        wt[2] = wt[1]
        wt[3] = wt[0]
    # inverse of weights
    [ai.__setitem__(k, 1./wt[k]) for k in range(0, mp+1)]
    # here gamma is different from classic compressible Euler eqns
    gamma = 3.
    gm1 = gamma - 1.

def init():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    xleft = -0.2
    xright = 1.2
    xlen = xright - xleft
    dx = xlen / nx
    cdx = 1 / dx
    dx2 = dx / 2
    x[0+sft] = xleft - 0.5 * dx
    x[-1+sft] = x[0+sft] - dx
    x[-2+sft] = x[-1+sft] - dx
    x[-3+sft] = x[-2+sft] - dx
    [x.__setitem__(i, x[i-1] + dx) for i in range(1+sft, nx+3+sft+1)]
    vleft = -4.5
    vright = -vleft
    vmax = abs(vleft)
    em = 3 * (vmax + 1)
    [v.__setitem__(j, (vright-vleft)/2*vq[j]+(vright+vleft)/2) for j in range(0, nvm+1)]

    rhol = 1.
    vell = 0.
    teml = 1. / 1.

    rhor = 0.125
    velr = 0.
    temr = 0.1 / 0.125

    for i in range(-3+sft, nx+3+sft+1):
        for k in range(0, mp+1):
            if i <= nx/2 + sft:
                u[k, i, 1] = rhol
                u[k, i, 2] = rhol * vell
                u[k, i, 3] = 0.5 * rhol * vell ** 2 + 0.5 * rhol * teml
            else:
                u[k, i, 1] = rhor
                u[k, i, 2] = rhor * velr
                u[k, i, 3] = 0.5 * rhor * velr ** 2 + 0.5 * rhor * temr

def setdt():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    dt = cflc * dx / max(em, abs(vleft))
    if (t + dt) > tfinal:
        dt = tfinal - t

    return dt

def bc_u():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    if i_bc == 1:
        for i in range(0, mp + 1):
            for k in range(4):
                for m in range(1, mn + 1):
                    u[i, -k + sft, m] = u[i, k + 1 + sft, m]
                    u[i, nx + k + 1 + sft, m] = u[i, nx - k + sft, m]
                u[i, -k + sft, 2] = -u[i, k + 1 + sft, 2]
                u[i, nx + k + 1 + sft, 2] = -u[i, nx - k + sft, 2]
                utem[i, -k + sft] = utem[i, k + 1 + sft]
                utem[i, nx + k + 1 + sft] = utem[i, nx - k + sft]

    elif i_bc == 2:
        for i in range(0, mp + 1):
            for k in range(4):
                for m in range(1, mn + 1):
                    u[i, -k + sft, m] = u[i, nx - k + sft, m]
                    u[i, nx + k + sft, m] = u[i, k + sft, m]
                utem[i, -k + sft] = utem[i, nx - k + sft]
                utem[i, nx + k + sft] = utem[i, k + sft]

    elif i_bc == 3:
        for i in range(0, mp + 1):
            for k in range(4):
                for m in range(1, mn + 1):
                    u[i, -k + sft, m] = u[i, 1 + sft, m]
                    u[i, nx + k + sft, m] = u[i, nx + sft, m]
                utem[i, -k + sft] = utem[i, 1 + sft]
                utem[i, nx + k + sft] = utem[i, nx + sft]

def bc_g():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    if i_bc == 1:
        for i in range(0, mp + 1):
            for k in range(4):
                for j in range(0, (nvm - 1) // 2 + 1):
                    # Outflow part
                    g[i, -k + sft, j] = g[i, 1 + sft, j]
                    g[i, nx + k + 1 + sft, nvm - j] = g[i, nx + sft, nvm - j]
                    # Reflective for the other part
                    g[i, -k + sft, nvm - j] = g[i, -k + sft, j]
                    g[i, nx + k + 1 + sft, j] = g[i, nx + k + 1 + sft, nvm - j]

    elif i_bc == 2:
        for i in range(0, mp + 1):
            for k in range(4):
                for j in range(0, nvm + 1):
                    g[i, -k + sft, j] = g[i, nx - k + sft, j]
                    g[i, nx + k + sft, j] = g[i, k + sft, j]

    elif i_bc == 3:
        for i in range(0, mp + 1):
            for k in range(4):
                for j in range(0, nvm + 1):
                    g[i, -k + sft, j] = g[i, 0 + sft, j]
                    g[i, nx + k + sft, j] = g[i, nx + sft, j]

def pn(xx, kk):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    pn = 1.
    for k in range(0, mp + 1):
        if k != kk:
            pn = pn * (xx - xp[k])/(xp[kk] - xp[k])
    return pn

def pnd(xx, kk):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    pnd = 0.
    if mp >= 1:
        for k1 in range(0, mp + 1):
            if k1 != kk:
                temp = 1.
                for k2 in range(0, mp + 1):
                    if k2 != k1 and k2 != kk:
                        temp *= (xx-xp[k2])/(xp[kk]-xp[k2])

                pnd += temp / (xp[kk]-xp[k1])

    return pnd

def poly(a, m, xx0):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    poly = 0.
    poly = sum(a[i] * pn(xx0, i) for i in range(0, m + 1))
    return poly

def uave(a):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    uave=0.
    uave = sum(wt[k] * a[k] for k in range(0, mp + 1))
    return uave

def sminmod3(xx, yy, zz):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    if abs(xx) <= 1. * dx ** 2:
        sminmod3 = xx
    else:
        sminmod3 = np.sign(xx) * max(0.0, min(abs(xx), yy * np.sign(xx), zz * np.sign(xx)))
    return sminmod3

def vm(vv, mm):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    if mm==1:
        vm = 1.
    elif mm==2:
        vm = vv
    elif mm==3:
        vm = vv ** 2 / 2.
    return vm

def ave(a):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    ave = 0.
    ave = sum(a[j] for j in range(1, nvm - 1 + 1))
    ave = ave + 0.5 * (a[0] + a[nvm])
    ave = ave * (vright - vleft) / nvm
    return ave

def gint(a, kk):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    vc = 0.
    if mp >= 1:
        vc = sum(wt[k] * a[k] * pnd(xp[k], kk) for k in range(0, mp + 1))

    return vc

def fint(a, kk, mm):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    uu = np.zeros(mnm + 1)
    vc = 0.
    if mp >= 1:
        for k in range(0, mp + 1):
            xx = xp[k]
            [uu.__setitem__(m, a[k, m]) for m in range(1, mn + 1)]
            if mm==1:
                ff = uu[2]
            elif mm==2:
                ff = 2. * uu[3]
            elif mm==3:
                ff = (uu[3] + 2. * (uu[3] - 0.5 * uu[2] ** 2 / uu[1])) * uu[2] / uu[1]

            vc += wt[k] * ff * pnd(xx, kk)

    return vc

def umint(a, kk):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    aa = np.zeros(md + 1)
    uu = np.zeros(mnm + 1)
    vc = 0.
    if mp >= 1:
        for k in range(0, mp + 1):
            xx = xp[k]
            [uu.__setitem__(m, a[k, m]) for m in range(1, mn + 1)]

            tem = abs((uu[3] - 0.5 * uu[2] ** 2 / uu[1]) / 0.5 / uu[1])
            vc += wt[k] * tem * pnd(xx, kk)

    return vc

def proj(a, uu, vv):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    b = np.zeros(nvm + 1)

    rho = uu[1]
    vel = uu[2] / uu[1]
    tem = abs(uu[3] / 0.5 / uu[1] - vel ** 2)

    vmu = rho / np.sqrt(2.0 * pi * tem) * np.exp(-(vv - vel) ** 2 / (2.0 * tem))
    [b.__setitem__(j, (v[j] - vel) * a[j]) for j in range(nvm + 1)]
    proj = ave(a) + (vv - vel) * ave(b) / tem

    [b.__setitem__(j, ((v[j] - vel) ** 2 / (2.0 * tem) - 0.5) * a[j]) for j in range(nvm + 1)]
    proj += ((vv - vel) ** 2 / (2.0 * tem) - 0.5) * 2.0 * ave(b)

    proj = proj / rho * vmu

    return proj

def update(io):
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    fluxg = np.zeros(ndm + 4)
    a = np.zeros(md + 1)
    b = np.zeros(nvm + 1)
    fluxu = np.zeros((ndm + 4, mnm + 1))
    aa = np.zeros((md + 1, mnm + 1))
    fluxmu = np.zeros(ndm + 4)
    fluxr = np.zeros(ndm + 4)
    um = np.zeros((ndm + 4, mnm + 1))
    eigl = np.zeros(mnm + 1)
    up = np.zeros((ndm + 4, mnm + 1))
    eigr = np.zeros(mnm + 1)
    gmass = np.zeros((md + 1, ndm + 4, nvm + 1))
    umass = np.zeros((md + 1, ndm + 4, nvm + 1))
    fm = np.zeros((ndm + 4, mnm + 1))
    fp = np.zeros((ndm + 4, mnm + 1))
    am = np.ones(mnm + 1) * 1e-15
    utp = np.zeros(mnm + 1)

    # Update u
    bc_u()
    bc_g()

    em = 1e-15
    for i in range(-1 + sft, nx + 1 + sft):
        for mm in range(1, mn + 1):
            for k in range(0, mp + 1):
                a[k] = u[k, i + 1, mm]
            up[i, mm] = poly(a, mp, -0.5)

        rhor = up[i, 1]
        velr = up[i, 2] / up[i, 1]
        temr = abs(up[i, 3] / 0.5 / up[i, 1] - velr ** 2)
        engr = up[i, 3]
        cvelr = np.sqrt(gamma * temr)
        eigr[1] = velr - cvelr
        eigr[2] = velr
        eigr[3] = velr + cvelr
        fp[i, 1] = up[i, 2]
        fp[i, 2] = 2 * engr
        fp[i, 3] = (engr + rhor * temr) * velr
        am[1] = max(am[1], abs(eigr[1]))
        am[2] = max(am[2], abs(eigr[2]))
        am[3] = max(am[3], abs(eigr[3]))

        for mm in range(1, mn + 1):
            for k in range(0, mp + 1):
                a[k] = u[k, i, mm]
            um[i, mm] = poly(a, mp, 0.5)

        rhol = um[i, 1]
        vell = um[i, 2] / um[i, 1]
        teml = abs(um[i, 3] / 0.5 / um[i, 1] - vell ** 2)
        engl = um[i, 3]
        cvell = np.sqrt(gamma * teml)
        eigl[1] = vell - cvell
        eigl[2] = vell
        eigl[3] = vell + cvell
        fm[i, 1] = um[i, 2]
        fm[i, 2] = 2 * engl
        fm[i, 3] = (engl + rhol * teml) * vell
        am[1] = max(am[1], abs(eigl[1]))
        am[2] = max(am[2], abs(eigl[2]))
        am[3] = max(am[3], abs(eigl[3]))

    em = max(am[1], am[3])

    for i in range(-1 + sft, nx + 1 + sft):
        for m in range(1, mn + 1):
            fluxu[i, m] = 0.5 * (fm[i, m] + fp[i, m] - em * (up[i, m] - um[i, m]))

    for m in range(1, mn + 1):
        # Flux of <vmg>: central
        for i in range(-1 + sft, nx + 1 + sft):
            if indk[i] == 0 or indk[i + 1] == 0:
                for k in range(0, mp + 1):
                    for j in range(nvm + 1):
                        b[j] = v[j] * vm(v[j], m) * g[k, i, j]
                    a[k] = ave(b)
                fluxg[i] = poly(a, mp, 0.5)

                for k in range(0, mp + 1):
                    for j in range(nvm + 1):
                        b[j] = v[j] * vm(v[j], m) * g[k, i + 1, j]
                    a[k] = ave(b)
                fluxg[i] = eps * 0.5 * (fluxg[i] + poly(a, mp, -0.5))

        for i in range(0 + sft, nx + 1 + sft):
            for k in range(0, mp + 1):
                for kk in range(mp + 1):
                    for j in range(nvm + 1):
                        b[j] = v[j] * vm(v[j], m) * g[kk, i, j]
                    a[kk] = eps * ave(b)
                gcell = gint(a, k)

                for mm in range(1, mn + 1):
                    for kk in range(mp + 1):
                        aa[kk, mm] = u[kk, i, mm]
                fcell = fint(aa, k, m)

                flxu[k, i, m, io - 1] = -fcell + fluxu[i, m] * pn(0.5, k) - fluxu[i - 1, m] * pn(-0.5, k)
                flxg[k, i, m, io - 1] = -gcell + fluxg[i] * pn(0.5, k) - fluxg[i - 1] * pn(-0.5, k)

                ffu = 0.0
                ffg = 0.0
                for is_ in range(1, io + 1):
                    ffu += rk_ex[io - 1, is_ - 1] * flxu[k, i, m, is_ - 1]
                    ffg += rk_ex[io - 1, is_ - 1] * flxg[k, i, m, is_ - 1]
                uc[k, i, m, io - 1] = u0[k, i, m] - dt * ai[k] * cdx * (ffu + ffg)

    # update g with newest U
    # note: U is recorded on each stage, which might not be necessary
    for m in range(1, mn + 1):
        for i in range(0 + sft, nx + 1 + sft):
            for k in range(0, mp + 1):
                u[k, i, m] = uc[k, i, m, io - 1]

    bc_u()

    # DG formulation of T_x: central flux
    for i in range(-1 + sft, nx + 1 + sft):
        for mm in range(1, mn + 1):
            [a.__setitem__(kk, u[kk, i + 1, mm]) for kk in range(0, mp + 1)]
            utp[mm] = poly(a, mp, -0.5)
        rho = utp[1]
        vel = utp[2] / utp[1]
        tem = abs(utp[3] / 0.5 / rho - vel ** 2)
        fluxmu[i] = tem

        for mm in range(1, mn + 1):
            [a.__setitem__(kk, u[kk, i, mm]) for kk in range(0, mp + 1)]
            utp[mm] = poly(a, mp, 0.5)
        rho = utp[1]
        vel = utp[2] / utp[1]
        tem = abs(utp[3] / 0.5 / rho - vel ** 2)
        fluxmu[i] = 0.5 * (tem + fluxmu[i])

    for i in range(0 + sft, nx + 1 + sft):
        for k in range(0, mp + 1):
            for mm in range(1, mn + 1):
                [aa.__setitem__((kk, mm), u[kk, i, mm]) for kk in range(0, mp + 1)]
                utp[mm] = u[k, i, mm]

            umcell = umint(aa, k)
            utem[k, i] = ai[k] * cdx * (-umcell + fluxmu[i] * pn(0.5, k) - fluxmu[i - 1] * pn(-0.5, k))

    # v loop
    for j in range(0, nvm + 1):
        # upwind flux for vg
        if v[j] >= 0.:
            for i in range(-1 + sft, nx + 1 + sft):
                [a.__setitem__(k, g[k, i, j]) for k in range(0, mp + 1)]
                fluxg[i] = eps * poly(a, mp, 0.5)
        else:
            for i in range(-1 + sft, nx + 1 + sft):
                [a.__setitem__(k, g[k, i + 1, j]) for k in range(0, mp + 1)]
                fluxg[i] = eps * poly(a, mp, -0.5)

        for i in range(0 + sft, nx + 1 + sft):
            for k in range(0, mp + 1):
                [a.__setitem__(kk, eps * g[kk, i, j]) for kk in range(0, mp + 1)]
                gcell = gint(a, k)
                gmass[k, i, j] = -gcell + fluxg[i] * pn(0.5, k) - fluxg[i - 1] * pn(-0.5, k)
                gmass[k, i, j] = v[j] * gmass[k, i, j]

        # (I - II)vMx = AMT_x / sqrt(T): scheme II
        for i in range(0 + sft, nx + 1 + sft):
            for k in range(0, mp + 1):
                for mm in range(1, mn + 1):
                    [aa.__setitem__((kk, mm), u[kk, i, mm]) for kk in range(0, mp + 1)]
                    utp[mm] = u[k, i, mm]
                rho = utp[1]
                vel = utp[2] / utp[1]
                tem = abs(utp[3] / 0.5 / rho - vel ** 2)

                coefA = ((v[j] - vel) ** 2 / (2. * tem) - 3. / 2.) * (v[j] - vel) / tem
                fmu = rho / np.sqrt(2. * pi * tem) * np.exp(-(v[j] - vel) ** 2 / (2. * tem))
                umass[k, i, j] = coefA * fmu * utem[k, i]

    # projection I - II on vg
    for i in range(0 + sft, nx + 1 + sft):
        for k in range(0, mp + 1):
            [utp.__setitem__(mm, u[k, i, mm]) for mm in range(1, mn + 1)]
            for j in range(0, nvm + 1):
                [b.__setitem__(jj, gmass[k, i, jj]) for jj in range(0, nvm + 1)]
                gave = proj(b, utp, v[j])
                flxmg[k, i, j, io - 1] = gmass[k, i, j] - gave
                flxmu[k, i, j, io - 1] = umass[k, i, j]

                ffu = 0.
                ffg = 0.
                for is_ in range(1, io + 1):
                    ffu = ffu + rk_im[io - 1, is_ - 1] * flxmu[k, i, j, is_ - 1]
                    ffg = ffg + rk_ex[io - 1, is_ - 1] * flxmg[k, i, j, is_ - 1]

                gg = 0.
                if io >= 2:
                    gg = sum(rk_im[io - 1, is_ - 1] * gc[k, i, j, is_ - 1] for is_ in range(1, io - 1 + 1))

                gc[k, i, j, io - 1] = eps / dt * g0[k, i, j] - (ai[k] * cdx * ffg + ffu + gg)
                gc[k, i, j, io - 1] = gc[k, i, j, io - 1] / (eps / dt + rk_im[io - 1, io - 1])
                g[k, i, j] = gc[k, i, j, io - 1]


def tvb_limiter():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    evr = np.zeros((ndm + 4, mnm + 1, mnm + 1))
    evl = np.zeros((ndm + 4, mnm + 1, mnm + 1))
    ind = np.zeros((ndm + 4, mnm + 1))
    ubar = np.zeros((ndm + 4, mnm + 1))
    dm = np.zeros(mnm + 1)
    dp = np.zeros(mnm + 1)
    dt0 = np.zeros(mnm + 1)
    a = np.zeros(md + 1)
    dtm = np.zeros(mnm + 1)
    dtp = np.zeros(mnm + 1)
    ddtm = np.zeros(mnm + 1)
    ddtp = np.zeros(mnm + 1)
    utx = np.zeros(mnm + 1)
    uk = np.zeros((md + 1, ndm + 4, mnm + 1))
    ux = np.zeros((ndm + 4, mnm + 1))
    um = np.zeros((ndm + 4, mnm + 1))
    up = np.zeros((ndm + 4, mnm + 1))

    # Example of boundary conditions and computation
    if mp >= 1:
        bc_u()
        for m in range(1, mn + 1):
            for i in range(-3 + sft, nx + 3 + 1 + sft):
                for k in range(0, mp + 1):
                    a[k] = u[k, i, m]

                um[i, m] = poly(a, mp, 0.5)
                up[i, m] = poly(a, mp, -0.5)
                ubar[i, m] = uave(a)
                uxx = 0.
                uxx = sum(a[k]*xp[k] for k in range(0, mp + 1))
                ux[i, m] = 12.0 * uxx

        # Compute characteristic field
        for i in range(0 + sft, nx + 1 + sft):
            rho = ubar[i, 1]
            vel = ubar[i, 2] / ubar[i, 1]
            tem = np.abs(ubar[i, 3] / 0.5 / rho - vel ** 2)
            eng = ubar[i, 3]

            vxm = vel
            hm = eng / rho + tem
            qm = 0.5 * vxm ** 2
            cm = np.sqrt(gm1 * abs(hm - qm))
            t0 = vxm * cm

            evr[i, 1, 1] = 1.0
            evr[i, 1, 2] = 1.0
            evr[i, 1, 3] = 1.0
            evr[i, 2, 1] = vxm - cm
            evr[i, 2, 2] = vxm
            evr[i, 2, 3] = vxm + cm
            evr[i, 3, 1] = hm - t0
            evr[i, 3, 2] = qm
            evr[i, 3, 3] = hm + t0

            rcm = 1.0 / cm
            b1 = gm1 * rcm ** 2
            b2 = qm * b1
            t0 = vxm * rcm
            t1 = b1 * vxm
            t2 = 0.5 * b1

            evl[i, 1, 1] = 0.5 * (b2 + t0)
            evl[i, 1, 2] = -0.5 * (t1 + rcm)
            evl[i, 1, 3] = t2
            evl[i, 2, 1] = 1.0 - b2
            evl[i, 2, 2] = t1
            evl[i, 2, 3] = -b1
            evl[i, 3, 1] = 0.5 * (b2 - t0)
            evl[i, 3, 2] = -0.5 * (t1 - rcm)
            evl[i, 3, 3] = t2

        # Correction for boundary values in cell i
        for i in range(0 + sft, nx + 1 + sft):
            for m in range(1, mn + 1):
                dm[m] = ubar[i, m] - ubar[i - 1, m]
                dp[m] = ubar[i + 1, m] - ubar[i, m]

            for m in range(1, mn + 1):
                dt0[m] = 0.0
                dtm[m] = 0.0
                dtp[m] = 0.0
                ddtm[m] = 0.0
                ddtp[m] = 0.0
                utx[m] = 0.0

                for mm in range(1, mn + 1):
                    dt0[m] += evl[i, m, mm] * ubar[i, mm]
                    dtm[m] += evl[i, m, mm] * dm[mm]
                    dtp[m] += evl[i, m, mm] * dp[mm]

                    ddtm[m] += evl[i, m, mm] * um[i, mm]
                    ddtp[m] += evl[i, m, mm] * up[i, mm]
                    utx[m] += evl[i, m, mm] * ux[i, mm]

                for k in range(0, mp + 1):
                    uk[k, i, m] = 0.0

                    for mm in range(1, mn + 1):
                        uk[k, i, m] += evl[i, m, mm] * u[k, i, mm]

            for m in range(1, mn + 1):
                dr = sminmod3(ddtm[m] - dt0[m], dtm[m], dtp[m])
                dl = sminmod3(dt0[m] - ddtp[m], dtm[m], dtp[m])
                ur = dt0[m] + dr
                ul = dt0[m] - dl
                if abs(ur - ddtm[m]) > tol or abs(ul - ddtp[m]) > tol:
                    ind[i, m] = 1

            for m in range(1, mn + 1):
                if ind[i, m] == 1:
                    uhx = sminmod3(utx[m], dtm[m] * 2.0, dtp[m] * 2.0)
                    for k in range(0, mp + 1):
                        uk[k, i, m] = dt0[m] + uhx * xp[k]

            for k in range(0, mp + 1):
                for m in range(1, mn + 1):
                    u[k, i, m] = 0.0
                    for mm in range(1, mn + 1):
                        u[k, i, m] += evr[i, m, mm] * uk[k, i, mm]

# 3rd order IMEX scheme
def DGIMEX3():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    bc_u()
    bc_g()
    tvb_limiter()

    indk = np.zeros(ndm + 4)

    u0 = u
    g0 = g

    update(1)

    for io in range(2, 4 + 1):
        tvb_limiter()
        update(io)


def saves():
    global u, g, u0, g0, utem, uc, gc, flxu, flxg, flxmu, flxmg, x, dx, cdx, dx2, v, vq, wq, vleft, vright
    global xq, cq, xp, wt, ai, rk_ex, rk_im, mp, mn, mo, i_bc, kcmax, kcount, nx
    global t, dt, tfinal, eps, pi, cflc, cfld, em, vmax, gamma, gm1, indk
    a = np.zeros(md + 1)
    uu = np.zeros(mnm + 1)
    b = np.zeros(nvm + 1)
    utp = np.zeros(mnm + 1)
    fluxmu = np.zeros(ndm + 4)
    aa = np.zeros((md + 1, mnm + 1))

    print(' ')
    print(f' time={t}, {kcount}')

    for i in range(-1 + sft, nx + 1 + sft):
        for mm in range(1, mnm + 1):
            [a.__setitem__(kk, u[kk, i + 1, mm]) for kk in range(0, mp + 1)]
            utp[mm] = poly(a, mp, -0.5)
        rho = utp[1]
        vel = utp[2] / utp[1]
        tem = utp[3] / 0.5 / utp[1] - vel ** 2
        fluxmu[i] = tem

        for mm in range(1, mnm + 1):
            [a.__setitem__(kk, u[kk, i, mm]) for kk in range(0, mp + 1)]
            utp[mm] = poly(a, mp, 0.5)
        rho = utp[1]
        vel = utp[2] / utp[1]
        tem = utp[3] / 0.5 / utp[1] - vel ** 2

        fluxmu[i] = 0.5 * (tem + fluxmu[i])

    for i in range(0 + sft, nx + 1 + sft):
        for k in range(0, mp + 1):
            for mm in range(1, mn + 1):
                [aa.__setitem__((kk, mm), u[kk, i, mm]) for kk in range(0, mp + 1)]
            umcell = umint(aa, k)
            utem[k, i] = ai[k] * cdx * (-umcell + fluxmu[i] * pn(0.5, k) - fluxmu[i - 1] * pn(-0.5, k))

    for i in range(1 + sft, nx + 1 + sft):
        for m in range(1, mn + 1):
            [a.__setitem__(k, u[k, i, m]) for k in range(mp + 1)]
            uu[m] = poly(a, mp, 0.0)

        with open('U.txt', 'a') as f_u:
            for k in range(mp + 1):
                f_u.write(
                    f"{t:.5e} {x[i] + xp[k] * dx:.5e} {u[k, i, 1]:.5e} {u[k, i, 2] / u[k, i, 1]:.5e} {(u[k, i, 3] - 0.5 * u[k, i, 2] ** 2 / u[k, i, 1]) / 0.5 / u[k, i, 1]:.5e}\n")

        with open('g.txt', 'a') as f_g:
            for k in range(mp + 1):
                for j in range(nvm + 1):
                    f_g.write(f"{t:.5e} {x[i] + xp[k] * dx:.5e} {v[j]:.5e} {g[k, i, j]:.5e}\n")


# ================= Program starts =================
initdata()
nx = 50

tbegin = time.time()
init()
dt = setdt()

# ============== Begin time evolution ==============

while t < tfinal - 1.e-15 and kcount < kcmax:
    print(kcount)
    ipause = 0
    if t >= tfinal - 1.e-15:
        break
    if kcount % 200 == 0 and kcount != 0:
        ipause = 1
        print(t, dt)

    DGIMEX3()

    t += dt
    kcount += 1
    if ipause == 1:
        saves()

tend = time.time()
print(' ')
print(t, kcount, tend - tbegin)

saves()
