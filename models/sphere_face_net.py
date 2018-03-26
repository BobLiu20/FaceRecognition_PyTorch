import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math


class MarginInnerProduct(nn.Module):
    def __init__(self, margin_params, **kwargs):
        super(MarginInnerProduct, self).__init__()
        #  args
        self.in_units = margin_params.get("feature_dim", 512)
        self.out_units = margin_params["class_num"]
        #  lambda parameter
        self.lamb_iter = margin_params.get("lamb_iter", 0)
        self.lamb_base = margin_params.get("lamb_base", 1500)
        self.lamb_gamma = margin_params.get("lamb_gamma", 0.01)
        self.lamb_power = margin_params.get("lamb_power", 1)
        self.lamb_min = margin_params.get("lamb_min", 10)
        #  margin type
        self.margin = margin_params.get("margin", 4)
        self.margin_cos = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]
        #  training parameter
        self.weight = Parameter(torch.Tensor(self.in_units, self.out_units))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, label):
        #  weight normalize
        w = self.weight
        w_norm = F.normalize(w, dim=1)
        # w.data[:] = w_norm.data
        #  cos_theta = x'w/|x|
        x_norm = x.norm(p=2, dim=1)
        output = x.mm(w)
        cos_theta = output / x_norm.view(-1, 1)
        cos_theta = cos_theta.clamp(-1, 1)
        #  cos_m_theta = cos(m * theta)
        cos_m_theta = self.margin_cos[self.margin](cos_theta)
        #  k
        theta = Variable(cos_theta.data.acos())
        k = (self.margin * theta / math.pi).floor()
        #  i=j is phi_theta and i!=j is cos_theta
        phi_theta = ((-1)**k) * cos_m_theta - 2 * k
        x_norm_phi_theta = x_norm.view(-1, 1) * phi_theta
        x_norm_cos_theta = x_norm.view(-1, 1) * cos_theta
        #  i=j index
        index = x_norm_cos_theta.data * 0.0
        index.scatter_(1, label.data.view(-1, 1), 1)
        # index = index.byte()
        index = Variable(index)
        #  output
        lamb = self.__get_lambda()
        output2 = output * (1.0 - index) + x_norm_phi_theta * index
        output3 = (output2 + lamb * output) / (1 + lamb)
        return output3

    def __get_lambda(self):
        self.lamb_iter += 1
        val = self.lamb_base * (1.0 + self.lamb_gamma * self.lamb_iter) ** (-self.lamb_power)
        val = max(self.lamb_min, val)
        if self.lamb_iter % 500 == 0:
            print ("Now lambda = {}".format(val))
        return val


class CNNResidualBlock(nn.Module):
    def __init__(self, in_out_cnn, in_out_res, num_residual, **kwargs):
        super(CNNResidualBlock, self).__init__()
        self.cnn = nn.Conv2d(*in_out_cnn, kernel_size=3, stride=2, padding=1)
        self.relu = nn.PReLU(in_out_cnn[1])
        self.residuals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(*in_out_res, kernel_size=3, stride=1, padding=1),
                nn.PReLU(in_out_res[1]),
                nn.Conv2d(*in_out_res, kernel_size=3, stride=1, padding=1),
                nn.PReLU(in_out_res[1])
            ) for _ in range(num_residual)
        ])

    def forward(self, x):
        x = self.relu(self.cnn(x))
        for m in self.residuals:
            x = x + m(x)
        return x


class SphereFaceNet(nn.Module):
    def __init__(self, model_params={}):
        super(SphereFaceNet, self).__init__()
        self.class_num = model_params["class_num"]
        self.feature_dim = model_params.get("feature_dim", 512)

        layer_type = model_params.get("layer_type", "20layer")
        image_size = model_params.get("image_size", 112)
        self.main_net = self.__main_net(layer_type, self.feature_dim, image_size)

        self.margin_fc = MarginInnerProduct(model_params)
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, x, label):
        x = self.main_net(x)
        x = self.margin_fc(x, label)
        x = self.ce_loss(x, label)
        return x

    def __main_net(self, layer_type, feature_dim, image_size):
        class Flatten(nn.Module):
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return x
        if image_size % 16 != 0:
            raise Exception("image size must be % 16 == 0")
        out_size = image_size / 16
        if layer_type == "20layer":
            model = nn.Sequential(
                CNNResidualBlock((3, 64), (64, 64), 1),
                CNNResidualBlock((64, 128), (128, 128), 2),
                CNNResidualBlock((128, 256), (256, 256), 4),
                CNNResidualBlock((256, 512), (512, 512), 1),
                Flatten(),
                nn.Linear(512 * out_size * out_size, feature_dim)
            )
        elif layer_type == "64layer":
            model = nn.Sequential(
                CNNResidualBlock((3, 64), (64, 64), 3),
                CNNResidualBlock((64, 128), (128, 128), 8),
                CNNResidualBlock((128, 256), (256, 256), 16),
                CNNResidualBlock((256, 512), (512, 512), 3),
                Flatten(),
                nn.Linear(512 * out_size * out_size, feature_dim)
            )
        else:
            raise Exception("Unsupport layer type. You can very easy to implement it.")
        return model


if __name__ == "__main__":
    params = {"in_units": 4, "out_units": 6,
              "lamb_iter": 0, "lamb_base": 1000,
              "lamb_gamma": 0.0001, "lamb_power": 1, "lamb_min": 10}
    test = MarginInnerProduct(params)

    x = Variable(torch.ones(2, 4).float())
    label = Variable(torch.ones(2).long())
    print test(x, label)
