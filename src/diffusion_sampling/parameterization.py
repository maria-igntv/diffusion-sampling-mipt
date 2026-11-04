"""Conversions between VP/DDPM and VE/EDM states; no network evaluation here.

VP: x = alpha*x0 + beta*epsilon, alpha**2 + beta**2 = 1.
VE: y = x0 + sigma*epsilon, sigma=beta/alpha, y=x/alpha.
The checkpoint predicts epsilon; sample/v conversions are exposed for theory/tests.
"""

import torch


def prediction_to_x0(sample, prediction, alpha, beta, prediction_type="epsilon"):
    if prediction_type == "epsilon":
        return (sample - beta * prediction) / alpha
    if prediction_type == "sample":
        return prediction
    if prediction_type == "v_prediction":
        return alpha * sample - beta * prediction
    raise ValueError(f"Unknown prediction type: {prediction_type}")


def prediction_to_epsilon(sample, prediction, alpha, beta, prediction_type="epsilon"):
    if prediction_type == "epsilon":
        return prediction
    if prediction_type == "sample":
        return (sample - alpha * prediction) / beta
    if prediction_type == "v_prediction":
        return alpha * prediction + beta * sample
    raise ValueError(f"Unknown prediction type: {prediction_type}")


def vp_to_ve(sample, sigma):
    return sample * torch.sqrt(1 + sigma.square())


def ve_to_vp(sample, sigma):
    return sample / torch.sqrt(1 + sigma.square())


def ve_epsilon_to_x0(sample, epsilon, sigma):
    return sample - sigma * epsilon
