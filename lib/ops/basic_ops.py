import torch


class Identity(torch.nn.Module):
    def forward(self, input):
        return input


class Example(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


class SegmentConsensus(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, consensus_type, dim):
        ctx.shape = input_tensor.size()
        ctx.consensus_type = consensus_type
        ctx.dim = dim

        if ctx.consensus_type == 'avg':
            output = input_tensor.mean(dim=ctx.dim, keepdim=True)
        elif ctx.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        # ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # output, = ctx.saved_tensors
        if ctx.consensus_type == 'avg':
            grad_in = grad_output.expand(ctx.shape) / float(ctx.shape[ctx.dim])
        elif ctx.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in, None, None


class _SegmentConsensus(torch.autograd.Function):

    def __init__(self, consensus_type, dim=1):
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, input_tensor):
        self.shape = input_tensor.size()
        if self.consensus_type == 'avg':
            output = input_tensor.mean(dim=self.dim, keepdim=True)
        elif self.consensus_type == 'identity':
            output = input_tensor
        else:
            output = None

        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = grad_output.expand(self.shape) / float(self.shape[self.dim])
        elif self.consensus_type == 'identity':
            grad_in = grad_output
        else:
            grad_in = None

        return grad_in


class ConsensusModule(torch.nn.Module):

    def __init__(self, consensus_type, dim=1):
        super(ConsensusModule, self).__init__()
        self.consensus_type = consensus_type if consensus_type != 'rnn' else 'identity'
        self.dim = dim

    def forward(self, input):
        return SegmentConsensus().apply(input, self.consensus_type, self.dim)
