import torch


class GeneralizedLogistic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, l, u, g):
        
        # save tensors for back prop
        ctx.save_for_backward(x, l, u, g)

        # set up equation
        power = (-g * x)
        exp_pow = torch.exp(power)
        s = 1.0 / (1.0 + exp_pow) # Txn

        # calculate y using the equation
        y = l + (u - l) * s

        return y

    @staticmethod
    def backward(ctx, dzdy):

        # retrive saved x, w, b tensors
        x, l, u, g = ctx.saved_tensors

        # caculate tensor s
        power = (-g * x)
        exp_pow = torch.exp(power)
        s = 1.0 / (1.0 + exp_pow)

        # use product rule to find grad for each component
        # I was running into dim errors on the scalars in xor tests, searched up that view makes a [a] scalar tensor as needed.
        dzdl = (dzdy * (1.0 - s)).sum().view(1) # scalar
        dzdu = (dzdy * s).sum().view(1) # scalar
        dzdx = dzdy * (u - l) * g * s * (1.0 - s) # tensor
        dzdg = (dzdy * (u - l) * x * s * (1.0 - s)).sum().view(1) # scalar

        return dzdx, dzdl, dzdu, dzdg
