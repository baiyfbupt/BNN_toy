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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=False)
        self.bconv1_1 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        #self.bconv1_2 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.trans1 = BinConv2d(512, 256, kernel_size=3, stride=1, padding=1)
       
        self.bconv2_1 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bconv2_2 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bconv2_3 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bconv2_4 = BinConv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.trans2 = BinConv2d(1024, 128, kernel_size=3, stride=1, padding=1, dropout=0.5)
 
        self.bconv3_1 = BinConv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bconv3_2 = BinConv2d(256, 256, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bconv3_3 = BinConv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bconv3_4 = BinConv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.trans3 = BinConv2d(1024, 128, kernel_size=3, stride=1, padding=1, dropout=0.5)
        
        self.bconv5_1 = BinConv2d(128, 2048, kernel_size=3, stride=1, padding=1, dropout=0.5)
        #self.bconv5_2 = BinConv2d(1024, 1024, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bn2 = nn.BatchNorm2d(2048, eps=1e-4, momentum=0.1, affine=False)
        self.conv2 = nn.Conv2d(2048, 10, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=4, stride=1, padding=0)



    def forward(self, x):
        out = self.conv1(x)
        bn_out = self.bn1(out)
        bconv1_1_out = self.bconv1_1(bn_out)
        #x = torch.cat((bn_out, bconv1_1_out), 1) 
        #x = bconv1_1_out
        #bconv1_2_out = self.bconv1_2(x)
        max1 = self.maxpool1(bconv1_1_out)
        #trans1 = self.trans1(max1)
        trans1 = max1 

        bconv2_1_out = self.bconv2_1(trans1)
        x = torch.cat((trans1, bconv2_1_out), 1)
        #x = bconv2_1_out
        bconv2_2_out = self.bconv2_2(x) 
        x = torch.cat((trans1, bconv2_1_out, bconv2_2_out), 1)
        #x = bconv2_2_out
        bconv2_3_out = self.bconv2_3(x)
        x = torch.cat((trans1, bconv2_1_out, bconv2_2_out, bconv2_3_out), 1)
        bconv2_4_out = self.bconv2_4(x)
        max2 = self.maxpool2(bconv2_4_out)
        trans2 = self.trans2(max2)
 
        bconv3_1_out = self.bconv3_1(trans2)
        x = torch.cat((trans2, bconv3_1_out), 1)
        #x = bconv3_1_out
        bconv3_2_out = self.bconv3_2(x) 
        x = torch.cat((trans2, bconv3_1_out, bconv3_2_out), 1)
        #x = bconv3_2_out
        bconv3_3_out = self.bconv3_3(x)
        x = torch.cat((trans2, bconv3_1_out, bconv3_2_out, bconv3_3_out), 1)
        bconv3_4_out = self.bconv3_4(x)
        max3 = self.maxpool3(bconv3_4_out) 
        max3 = self.trans3(max3)


        bconv5_1_out = self.bconv5_1(max3)
        #x = torch.cat((max3, bconv4_1_out), 1)
        #x = bconv4_1_out
        #bconv4_2_out = self.bconv4_2(x)
        bconv5_2_out = bconv5_1_out
        out = self.avgpool(self.relu(self.conv2(self.bn2(bconv5_2_out))))
        out = out.view(out.size(0), 10) 
        return out
