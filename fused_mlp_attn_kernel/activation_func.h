#define SIGMOID 0
#define RELU 1
#define TANH 2

// Generic activation function.
template <int activation_type, typename scalar_t>
__device__ scalar_t activation_func(scalar_t logit);

// Derivate of generic activation function.
template <int activation_type, typename scalar_t>
__device__ scalar_t activation_derivative(scalar_t logit);

// Sigmoid for floats.
template <>
__device__ float activation_func<SIGMOID, float>(float logit) {
    return 1.0f / (1.0f + expf(-logit));
}

// Sigmoid for doubles.
template <>
__device__ double activation_func<SIGMOID, double>(double logit) {
    return 1.0 / (1.0 + exp(-logit));
}

// Derivate of sigmoid for floats. Easy.
template <>
__device__ float activation_derivative<SIGMOID, float>(float logit) {
    float prob = activation_func<0, float>(logit);
    return prob * (1.0f - prob);
}

// Derivate of sigmoid for dobules. Easy.
template <>
__device__ double activation_derivative<SIGMOID, double>(double logit) {
    double prob = activation_func<0, double>(logit);
    return prob * (1.0 - prob);
}

// ReLU for floats
template <>
__device__ float activation_func<RELU, float>(float logit) {
    return fmaxf(0.0f, logit);
}

// ReLU for doubles
template <>
__device__ double activation_func<RELU, double>(double logit) {
    return fmax(0.0, logit);
}

// Derivate of ReLU for floats.
template <>
__device__ float activation_derivative<RELU, float>(float logit) {
    return logit > 0.0f ? 1.0f : 0.0f;
}

// Derivate of ReLU for doubles.
template <>
__device__ double activation_derivative<RELU, double>(double logit) {
    return logit > 0.0 ? 1.0 : 0.0;
}

// Tanh for floats.
template <>
__device__ float activation_func<TANH, float>(float logit) {
    return tanhf(logit);
}

// Tanh for doubles.
template <>
__device__ double activation_func<TANH, double>(double logit) {
    return tanh(logit);
}

// Derivate of Tanh for floats.
template <>
__device__ float activation_derivative<TANH, float>(float logit) {
    float prob = activation_func<TANH, float>(logit);
    return 1.0f - prob * prob;
}

// Derivate of Tanh for doubles.
template <>
__device__ double activation_derivative<TANH, double>(double logit) {
    double prob = activation_func<TANH, double>(logit);
    return 1.0 - prob * prob;
}