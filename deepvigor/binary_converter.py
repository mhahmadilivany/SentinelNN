#Manipulated and improved version of binary converter on: https://github.com/KarenUllrich/pytorch-binary-converter

import torch


def bit2float(b, device=torch.device('cpu'), num_e_bits=8, num_m_bits=23, bias=127.):
    dtype = torch.float32

    s = torch.index_select(b, -1, torch.arange(0, 1, device=device))
    e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits, device=device))
    m = torch.index_select(b, -1, torch.arange(1 + num_e_bits, 1 + num_e_bits + num_m_bits, device=device))
    
    ## SIGN BIT    
    out = (torch.pow(-(torch.ones(1, device=device)), s)).squeeze(-1).type(dtype)
    
    ## EXPONENT BIT
    exponents = -torch.arange(-(num_e_bits - 1.), 1., device=device)
    exponents = exponents.repeat(b.shape[:-1] + (1,))
    e_decimal = torch.sum(e * torch.pow(torch.ones(1, device=device) * 2, exponents), dim=-1) - bias
    out *= torch.pow(torch.ones(1, device=device) * 2, e_decimal)
    
    ## MANTISSA
    matissa = (torch.pow((torch.ones(1, device=device) * 2), -torch.arange(1., num_m_bits + 1., device=device))).repeat(m.shape[:-1] + (1,))
    
    out *= 1. + torch.sum(m * matissa, dim=-1)
    out = out.detach()
    
    return out


def float2bit(f, device=torch.device('cpu'), num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
    ## SIGN BIT
    s = torch.sign(f)
    f = f * s
    
    # turn sign into sign-bit
    s = (s * (-1) + 1.) * 0.5
    s = s.unsqueeze(-1)
    s[s == 0.5] = 0

    ## EXPONENT BIT
    f[f==0] = 1
    e_scientific = torch.floor(torch.log2(f))

    e_decimal = e_scientific + bias
    e = integer2bit(e_decimal, num_bits=num_e_bits, device=device)

    ## MANTISSA
    int_precision = 256
    m1 = integer2bit(f - f % 1, num_bits=int_precision, device=device)
    m2 = remainder2bit(f % 1, num_bits=bias, device=device)
    m = torch.cat([m1, m2], dim=-1)
    dtype = f.type()
    
    idx = torch.arange(num_m_bits, device=device).unsqueeze(0).type(dtype) + (float(int_precision) - e_scientific).unsqueeze(-1)
    idx = idx.long()
    
    m = torch.gather(m, dim=-1, index=idx)
    
    out = torch.cat([s, e, m], dim=-1).type(dtype)
    out = out.detach()

    return out


def remainder2bit(remainder, num_bits=127, device=torch.device('cpu')):
    dtype = remainder.type()
    exponent_bits = torch.arange(num_bits, device=device).type(dtype)
    exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
    out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
    return torch.floor(2 * out)


def integer2bit(integer, num_bits=127, device=torch.device('cpu')):
    dtype = integer.type()
    exponent_bits = -torch.arange(-(num_bits - 1), 1, device=device).type(dtype)
    exponent_bits = exponent_bits.repeat(integer.shape + (1,))
    out = integer.unsqueeze(-1) / 2 ** exponent_bits
    return (out - (out % 1)) % 2
