import numpy as np
from bm3d import bm3d_deblurring

def psf2otf(psf, shape):
    inshape = psf.shape
    temp = np.zeros(shape)
    temp[0:inshape[0],0:inshape[1]] = psf
    psf = temp
    for axis, axis_size in enumerate(inshape):
        psf = np.roll(psf, - int(axis_size / 2), axis=axis)
    otf = np.fft.fft2(psf)
    return otf

def ForwardDx(U):
    return np.roll(U, -1, axis=1) - U

def ForwardDy(U):
    return np.roll(U, -1, axis=0) - U

def Dive(X, Y):
    DtXY = np.roll(X, 1, axis=1) - X
    DtXY = DtXY + np.roll(Y, 1, axis=0) - Y
    return DtXY

def KtF(k_otf, im_b):
    return np.real( np.fft.ifft2( np.conjugate(k_otf)*np.fft.fft2(im_b)))


def generate_dx_dy():
    dx = np.array([[ 1, -1]], dtype='float32')    

    dy = np.array([[ 1], 
                   [ -1]], dtype='float32')
    return dx, dy

def compute_params(k_s, dx, dy, I):
    # Some Parameter Computing
    k_otf  = psf2otf(k_s, (I.shape[0], I.shape[1]))
    dx_otf = psf2otf(dx,  (I.shape[0], I.shape[1]))
    dy_otf = psf2otf(dy,  (I.shape[0], I.shape[1]))
    Lam = np.zeros((I.shape[0], I.shape[1]))
    I_dx = ForwardDx(I)
    I_dy = ForwardDy(I)
    
    return k_otf, dx_otf, dy_otf, Lam, I_dx, I_dy
    
def FTL1_v4(I, im_b, k_s, Ind1, para, mu, mini_iter, relchg_o = 1):
    """???
    Args:
        I: ?
        im_b: ?
        relchg_o: ?
    
    Returns:
        
    
    """
    dx, dy = generate_dx_dy()
    Ind = (im_b > 0.9)|Ind1
    
    k_otf, dx_otf, dy_otf, Lam, I_dx, I_dy = compute_params(k_s, dx, dy, I)
    
    down = (para.beta2/para.beta1)* np.absolute (k_otf)**2 +( np.absolute(dx_otf)**2 +  np.absolute(dy_otf)**2) + para.Guide/para.beta1
    FFT_Guide_image = np.fft.fft2(I)
    
    temp = np.real( np.fft.ifft2( k_otf*np.fft.fft2(I)))
    Ind2 = temp >0.9
    res = temp - im_b
    del temp
    Ind3 = Ind2|Ind
    res[Ind3] = 0
    fftKtB = np.conjugate(k_otf)*np.fft.fft2(im_b)
    
    Lam1, Lam2, Lam3 = Lam.copy(), Lam.copy(), Lam.copy()
    for i in range(para.max_iter):
        
        Z1 = I_dx+Lam1/para.beta1
        Z2 = I_dy+Lam2/para.beta1
        Z3 = res + Lam3/para.beta2
        
        V = np.sqrt( Z1**2 + Z2**2)
        V[V==0]=1
        
        # Shrinkage Step
        V = np.maximum(V-1/para.beta1,0)/V
        Y1 = Z1*V
        Y2 = Z2*V
        z = np.maximum(np.abs(Z3)-mu/para.beta2,0)*np.sign(Z3)
        #X-subprolem
        I_pre = I
        
        Temp = (z*para.beta2-Lam3)/para.beta1
        Temp = np.conj(k_otf)*np.fft.fft2(Temp)
        up = ( np.fft.fft2(Dive(Y1 -Lam1/para.beta1,Y2-Lam2/para.beta1)) + Temp + (para.beta2/para.beta1)*fftKtB) + (para.Guide/para.beta1)*FFT_Guide_image
        I = np.real(np.fft.ifft2(up/down))
        relchg = np.linalg.norm(I-I_pre)/np.linalg.norm(I)
        print(relchg, end = '\r')
        if (relchg_o < relchg) & (i>=80):
            I_pre[I_pre>1] = 1
            return I_pre
        if (relchg < para.relchg_cond) & (i>=mini_iter):
            I[I>1]=1
            return I
        relchg_o = relchg
        I_dx = ForwardDx(I)
        I_dy = ForwardDy(I)
        temp = np.real( np.fft.ifft2( k_otf*np.fft.fft2(I)))
        
        Ind2 = temp >0.9
        res = temp - im_b
        del temp
        Ind3 = Ind|Ind2
        res[Ind3] = 0
        Lam1 = Lam1 - para.gamma*para.beta1*(Y1-I_dx)
        Lam2 = Lam2 - para.gamma*para.beta1*(Y2-I_dy)
        Lam3 = Lam3 - para.gamma*para.beta2*(z - res)
    I[I>1]=1
    return I

def FTL2(I, kernel_est, mu = 10000, relchg_cond = 1e-3, beta = 10, gamma = 1.618,max_iter = 100):
    """???
    Args:
        relchg_cond: ?
    
    Returns:
    
    """
        
    dx, dy = generate_dx_dy()
    k_otf, dx_otf, dy_otf, Lam, I_dx, I_dy = compute_params(kernel_est, dx, dy, I)
    I_KtF = KtF(k_otf, I)
    Lam1, Lam2 = Lam.copy(), Lam.copy()
    for i in range(max_iter):
        # Shrinkage Step
        Z1 = I_dx+Lam1/beta
        Z2 = I_dy+Lam2/beta
        V = np.sqrt( Z1**2 + Z2**2)
        V[V==0]=1
        V = np.maximum(V-1/beta,0)/V
        Y1 = Z1*V
        Y2 = Z2*V
        #X-subprolem
        I_pre = I.copy()
        up = np.fft.fft2( ( mu*I_KtF - Dive(Lam1, Lam2))/beta + Dive(Y1,Y2) )
        down = (mu/beta)*np.absolute(k_otf)**2 + ( np.absolute(dx_otf)**2 +  np.absolute(dy_otf)**2)
        I = np.real(np.fft.ifft2(up/down))
        relchg = np.linalg.norm(I-I_pre)/np.linalg.norm(I)
        if relchg < relchg_cond:
            return I
        I_dx = ForwardDx(I)
        I_dy = ForwardDy(I)
        Lam1 = Lam1-gamma*beta*(Y1-I_dx)
        Lam2 = Lam2-gamma*beta*(Y2-I_dy)

def run_bm3d_deblurring(blur_img, sigma, blur_kernel):

    """Run bm3d deblurring with data-shape checking

    Parameters:
    
        blur_image : np.array

        sigma : float
            Noise standard deviation

        blur_kernel : np.array
        
        regularization_alpha_ri: number
            larger -> smoother

    Returns:
    
        out : np.array
            deblurred image
            
    Examples:
        >>>  run_bm3d_deblurring(blur_img, sigma, blur_kernel)

    """
    
    result_dict = {}
    blur_img = np.expand_dims(blur_img, axis = 2)
    y_est = bm3d_deblurring(blur_img, sigma, blur_kernel)
    result = np.clip(y_est, 0, 1)
    result_dict['result_wo_clip'] = y_est
    result_dict['result_w_clip'] = result
    return result_dict
