#include "normalization.hpp"

RMSNormImpl::RMSNormImpl(int d, double eps) : eps(eps)
{
    w = register_parameter("w", torch::ones(d));
}

torch::Tensor RMSNormImpl::forward(torch::Tensor x)
{
    // x * rsqrt(mean(x^2) + eps) * weight
    return x * torch::rsqrt(x.pow(2).mean({-1}, true) + eps) * w;
}
