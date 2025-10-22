import torch

class MeanSquaredError(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):

        # save the tensors for the backprop
        ctx.save_for_backward(x1, x2)
        
        # get shape of tensors
        T, n = x1.shape

        # calculate total squared difference between x1 and x2 
        total_sum = 0.0
        for t in range(T):

            # calculate the inner sum for each t
            inner_sum = 0.0
            for i in range(n):
                diff = x1[t, i] - x2[t, i]
                inner_sum += diff ** 2
            
            total_sum += inner_sum

        # average the total squared difference
        y = total_sum/(T*n)

        return y

    @staticmethod
    def backward(ctx, dzdy):

        # retrieve saved tensors
        x1, x2 = ctx.saved_tensors

        # extract shape of x1 (same as x2)
        T, n = x1.shape

        # calculate each grad
        dzdx1 = dzdy * (((2 * (x1-x2))/ (T * n))) 
        dzdx2 = dzdy * (((-2 * (x1-x2))/ (T * n))) 

        return dzdx1, dzdx2