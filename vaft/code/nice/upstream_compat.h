#pragma once

// NICE 7ad1ea8 calls a global legendre(n, x), while modern standard-library
// headers only expose std::legendre.  Force-include this compatibility shim
// when building that pinned revision with AppleClang.
inline double legendre(unsigned int n, double x) {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double p0 = 1.0;
    double p1 = x;
    for (unsigned int k = 2; k <= n; ++k) {
        const double p = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / k;
        p0 = p1;
        p1 = p;
    }
    return p1;
}
