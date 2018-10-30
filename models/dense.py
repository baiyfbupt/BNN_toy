import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-4, momentum=0.1, affine=False)
        self.conv1 = BinConv2d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-4, momentum=0.1, affine=False)
        self.conv1 = BinConv2d(nChannels, nOutChannels, kernel_size=1, bias=False)
  
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet(nn.Module):
  def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
    super(DenseNet, self).__init__()

    nDenseBlocks = int( (depth-4) / 3 )

    nChannels = 2*growthRate
    self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1, bias=False)

    self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans1 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate
    nOutChannels = int(math.floor(nChannels*reduction))
    self.trans2 = Transition(nChannels, nOutChannels)

    nChannels = nOutChannels
    self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
    nChannels += nDenseBlocks*growthRate

    self.bn1 = nn.BatchNorm2d(nChannels, eps=1e-4, momentum=0.1, affine=False)
    self.fc = nn.Linear(nChannels, nClasses)

    L2_NORM = 0
    if L2_NORM:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
              m.weight.data.fill_(1)
              m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)
  
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = out.view(out.size(0), 10)
        #out = self.fc(out)
        return out

