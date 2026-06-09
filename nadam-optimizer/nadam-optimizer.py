import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Write code here
    m_t = [beta1*m_i + (1-beta1)*grad_i for m_i, grad_i in zip(m,grad)]
    v_t = [beta2*v_i + (1-beta2)*grad_i**2 for v_i, grad_i in zip(v,grad)]

    return ([w_i - lr*((beta1*m_i+(1-beta1)*grad_i)/(np.sqrt(v_i)+eps)) for w_i,m_i,v_i,grad_i in zip(w,m_t,v_t,grad)],m_t,v_t)