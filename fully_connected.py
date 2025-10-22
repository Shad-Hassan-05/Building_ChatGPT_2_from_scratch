import torch

class FullyConnected(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):

        # save the tensors for backwards 
        ctx.save_for_backward(x, w, b)

        # calculate y with fully connected formula 
        y = (x @ w) + b
        
        return y

    @staticmethod
    def backward(ctx, dz_dy):

        # retrive saved x, w, b tensors
        x, w, b = ctx.saved_tensors

        # use product rule to find grad for each component
        dzdx = dz_dy @ w.t() 
        dzdw = x.t() @ dz_dy
        dzdb = dz_dy.sum(dim=0) # sum the rows to fit the dimension

        return dzdx, dzdw, dzdb