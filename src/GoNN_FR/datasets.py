import torch
from torch.utils import data

""" XOR """
# dataset generator
class XorDataset(data.Dataset):
    """Create dataset for Xor learning."""

    def __init__(self, nsample=1000, test=False, discrete=False, range=(0,1)):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        self.classes = ["0", "1"]
        # self.data = torch.rand(self.nsample, 2)
        a, b = range
        if test:
            if discrete:
                self.data = torch.tensor(
                    [[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float)
                self.nsample = 4
            else:
                self.nsample //= 10
                self.data = torch.rand((self.nsample, 2)) * (b - a) + a
        else:
            if discrete:
                self.data = torch.bernoulli(
                    torch.ones((self.nsample, 2)) * 0.5)
            else:
                self.data = torch.rand((self.nsample, 2)) * (b - a) + a
        self.targets = torch.logical_xor(torch.round(self.data)[...,0], torch.round(self.data)[...,1]).long()

    def __getitem__(self, index):
        """Get a data point."""
        # assert index < self.nsample, "The index must be less than the number of samples."
        inp, lab = self.data[index], self.targets[index]
        return inp, lab

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


class OrDataset(data.Dataset):
    """Create dataset for Or learning."""

    def __init__(self, nsample=1000, test=False):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        self.classes = ["0", "1"]
        # self.data = torch.rand(self.nsample, 2)
        if test:
            self.data = torch.tensor([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=torch.float)
            self.nsample = 4
        else:
            self.data = torch.bernoulli(
                torch.ones((self.nsample, 2)) * 0.5)
        # self.targets = torch.logical_or(*torch.round(self.data)).type(torch.float)
        self.targets = torch.logical_or(torch.round(self.data)[...,0], torch.round(self.data)[...,1]).long()

    def __getitem__(self, index):
        """Get a data point."""
        # assert index < self.nsample, "The index must be less than the number of samples."
        inp, lab = self.data[index], self.targets[index]
        return inp, lab

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


""" XOR3D """
# dataset generator
class Xor3dDataset(data.Dataset):
    """Create dataset for 3D Xor learning."""

    def __init__(self, nsample=1000, test=False, discrete=True, range=(0,1)):
        """Init the dataset
        :returns: TODO

        """
        data.Dataset.__init__(self)
        self.nsample = nsample
        self.test = test
        # self.input_vars = torch.rand(self.nsample, 2)
        a, b = range
        if test:
            if discrete:
                self.input_vars = torch.cartesian_prod(
                    torch.tensor([a,b]), torch.tensor([a,b]), torch.tensor([a,b]),
                    ).float()
                self.nsample = len(self.input_vars)
            else:
                self.nsample //= 10
                self.input_vars = torch.rand((self.nsample, 3)) * (b - a) + a
        else:
            if discrete:
                self.input_vars = torch.bernoulli(
                    torch.ones((self.nsample, 3)) * 0.5)
            else:
                self.input_vars = torch.rand((self.nsample, 3)) * (b - a) + a

    def __getitem__(self, index):
        """Get a data point."""
        # assert index <= self.nsample, "The index must be less than the number of samples."
        inp = self.input_vars[index]
        return inp, torch.logical_xor(torch.round(inp[0]), torch.logical_xor(*torch.round(inp[1:]))).long() 

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample


""" CIRCLE """
# dataset generator
class CircleDataset(data.Dataset):
    """Circle dataset with n classes."""
    
    
    def __init__(self, nsample=1000, test=False, nclasses=2, noise=False):
        """Init the dataset with [nclasses] classes."""
        
        data.Dataset.__init__(self)
        self.nsample = (nsample // nclasses + 1) * nclasses  # make it a multiple of nclasses
        print(f"Warning: you asked for {nsample} samples with {nclasses} classes.\nTo have the same amount of samples per class, the new number of samples is {self.nsample}.")
        self.test = test
        self.nclasses = nclasses
        self.classes = [str(i) for i in range(nclasses)]
        # if test:
        #     self.nsample //= 10
        t = [(torch.rand((self.nsample // nclasses)) + k) / nclasses for k in range(nclasses)]
        self.data = torch.cat([
            torch.stack(
                (torch.cos(2 * torch.pi * t_k),
                 torch.sin(2 * torch.pi * t_k)),
                dim=-1)
            for t_k in t], dim=0)

        if noise:
            for i, p in enumerate(self.data):
                self.data[i] = p * (torch.randn(1) * 0.01 + 1)

        self.targets = torch.arange(0, self.nsample) // (self.nsample // self.nclasses)
        self.targets = self.targets.long()
    

    def __getitem__(self, index):
        """Get a data point."""
        # assert index <= self.nsample, "The index must be less than the number of samples."
        inp = self.data[index]
        lab = self.targets[index]
        return inp, lab

    def __len__(self):
        """Return len of the dataset."""
        return self.nsample

# CD = CircleDataset(100, nclasses=6, noise=True)
# import matplotlib.pyplot as plt
# colors = ['red', 'blue', 'yellow', 'black', 'green', 'orange']
# print(len(CD))
# for p, c in CD:
#     plt.plot(p[0], p[1], "o" ,color=colors[c])
#
# plt.show()
