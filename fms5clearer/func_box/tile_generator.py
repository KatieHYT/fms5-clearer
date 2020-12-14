import numpy as np
import math 
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def _tile_generater_two_di_two_no_direct(Ori_Img, m, n):
    ind_i = np.zeros(((n-2)*(m-2)*5,),dtype=int)
    ind_j = np.empty_like(ind_i)
    ind_k = np.empty_like(ind_i)
    ind_i_l = np.zeros((2*(m-2),),dtype=int)
    ind_j_l = np.empty_like(ind_i_l)
    ind_k_l = np.empty_like(ind_i_l)
    ind_i_r = np.empty_like(ind_i_l)
    ind_j_r = np.empty_like(ind_i_l)
    ind_k_r = np.empty_like(ind_i_l)
    ind_i_t = np.zeros(((n-2)*4,),dtype=int)
    ind_j_t = np.empty_like(ind_i_t)
    ind_k_t = np.empty_like(ind_i_t)
    ind_i_b = np.empty_like(ind_i_t)
    ind_j_b = np.empty_like(ind_i_t)
    ind_k_b = np.empty_like(ind_i_t)
    [strat, strat_l,strat_r,start_t,start_b] = [0,0,0,0,0]
    for i in range(1,n-1):
        ind_i_t[start_t:start_t+4] = [i,i,i,i]
        ind_j_t[start_t:start_t+4] = [i-1,i,i+1,i+n]
        ind_k_t[start_t:start_t+4] = [1,-4,1,1]
        start_t = start_t+4
    for i in range((m-1)*n+1,m*n-1):
        ind_i_b[start_b:start_b+4] = [i,i,i,i]
        ind_j_b[start_b:start_b+4] = [i-n,i-1,i,i+1]
        ind_k_b[start_b:start_b+4] = [1,1,-4,1]
        start_b = start_b+4
    for i in range(n,(m-1)*n):
        if (i)%n == 0:
            ind_i_l[strat_l:strat_l+2] = [i,i]
            ind_j_l[strat_l:strat_l+2] = [i,i+1]
            ind_k_l[strat_l:strat_l+2] = [-1,1]
            strat_l = strat_l+2
            continue;
        if (i)%n == n-1:
            ind_i_r[strat_r:strat_r+2] = [i,i]
            ind_j_r[strat_r:strat_r+2] = [i-1,i]
            ind_k_r[strat_r:strat_r+2] = [-1,1]
            strat_r = strat_r+2;
            continue;
        ind_i[strat:strat+5] = [i,i,i,i,i]
        ind_j[strat:strat+5] = [i-n,i-1,i,i+1,i+n]
        ind_k[strat:strat+5] = [1,1,-4,1,1]
        strat = strat+ 5; 
    t_ind_i = np.concatenate((ind_i,ind_i_l,ind_i_r,ind_i_t,ind_i_b))
    t_ind_j = np.concatenate((ind_j,ind_j_l,ind_j_r,ind_j_t,ind_j_b))
    t_ind_k = np.concatenate((ind_k,ind_k_l,ind_k_r,ind_k_t,ind_k_b))
    A=csc_matrix((t_ind_k, (t_ind_i, t_ind_j)),shape=(m*n,m*n))

    total_i = range(0,m*n)
    top = range(0,n)
    botton = range(n*(m-1),m*n)
    di_i = np.union1d(top,botton);
    di_inner_i = np.setdiff1d(total_i,di_i)
    
    Temp = np.zeros((m,n))
    Temp[m-1,:] = Ori_Img[0,:];
    Temp[0,:] = Ori_Img[Ori_Img.shape[0]-1,:];
    
    F = A[:,di_i].dot(np.concatenate( (Temp[0,:],Temp[m-1,:] )))
    x0=np.reshape(Temp.T,m*n,1)    
    
    x0[di_inner_i] = spsolve(A[di_inner_i][:,di_inner_i],
      -F[di_inner_i],use_umfpack=True)

    Tile_img = x0.reshape(m,n)
    return Tile_img


def _tile_generater_four_di_direct(v_r, v_l, v_b, v_t):        
    m = len(v_r);
    n = len(v_b);
    Temp = np.zeros((m,n));    
    Temp[:,0] = v_l;Temp[:,n-1] = v_r;Temp[0,:] = v_t;Temp[m-1,:] = v_b;
    
    i = range(0,m*n)
    k1 = -4*np.ones((1,m*n))
    k2 = np.ones((1,m*n-1))
    k3 = np.ones((1,m*n-n))
    is1 = i[0:m*n-1]
    is2 = i[1:m*n]
    is3 = i[0:m*n-n]
    is4 = i[0+n:m*n]
    
    total_i = np.concatenate((i,is1,is2,is3,is4),axis=0)
    total_j = np.concatenate((i,is2,is1,is4,is3),axis=0)
    total_k = np.concatenate((k1,k2,k2,k3,k3),axis=1)
    A =csc_matrix((total_k[0,:], (total_i,total_j)),shape=(m*n,m*n))
    
    total_i = range(0,m*n)
    top = range(0,n)
    botton = range(n*(m-1),m*n)
    left = np.arange(0,n*(m-1)+1,n)
    right = np.arange(n-1,m*n,n)
    out_i = np.union1d(np.union1d(np.union1d(top,botton),left),right)
    inner_i = np.setdiff1d(total_i,out_i);
    
    x0 = np.reshape(Temp.T,m*n,1);    
    F = -A[:,out_i].dot(x0[out_i])
    
    x0[inner_i] = spsolve(A[inner_i][:,inner_i],F[inner_i],use_umfpack=True)
        
    Tile_img = x0.reshape(m,n);
    return Tile_img

def tile_generator_direct(Ori_Img):
    m, n = Ori_Img.shape
    mk = math.ceil(math.log2(m))
    nk = math.ceil(math.log2(n))
    cmk = 2**mk; cnk = 2**nk;
    dm = cmk-m; dn = cnk-n;
    center_m = math.floor(dm/2);
    center_n = math.floor(dn/2);
    Tile_img1 = _tile_generater_two_di_two_no_direct(Ori_Img, dm, n)
    Tile_img2 = _tile_generater_two_di_two_no_direct(Ori_Img.T, dn, m).T
    Tile_img3 = _tile_generater_four_di_direct(Tile_img1[:,0],
                                             Tile_img1[:, Tile_img1.shape[1]-1],
                                             Tile_img2[0, :],
                                             Tile_img2[Tile_img2.shape[0]-1, :])
    Tile_img = np.concatenate((
            np.concatenate((
                    Tile_img3[center_m::, center_n::],
                    Tile_img1[center_m::, :],
                    Tile_img3[center_m::, 0:center_n]), axis=1),
            np.concatenate((
                    Tile_img2[:, center_n:Tile_img2.shape[1]],
                    Ori_Img,
                    Tile_img2[:, 0:center_n]), axis=1),
            np.concatenate((
                    Tile_img3[0:center_m, center_n:Tile_img3.shape[1]],
                    Tile_img1[0:center_m, :],
                    Tile_img3[0:center_m, 0:center_n]), axis=1)
            ),axis=0)
    return Tile_img, dm, dn

