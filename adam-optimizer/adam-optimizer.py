import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    
    m = [beta1*i + (1-beta1)*g for i,g in zip(m,grad)]
    v = [beta2*i + (1-beta2)*g**2 for i,g in zip(v,grad)]
    
    m_bias = [i/(1-beta1**t) for i in m]
    v_bias = [i/(1-beta2**t) for i in v]

    param = [i - lr*(m_i/(np.sqrt(v_i) + eps)) for i,m_i,v_i in zip(param,m_bias,v_bias)]
    return (param,m,v)