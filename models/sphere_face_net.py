import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
import copy


class MarginInnerProduct(nn.Module):
    def __init__(self, margin_params, parallel_mode, common_dict, **kwargs):
        super(MarginInnerProduct, self).__init__()
        self.common_dict = common_dict
        self.parallel_mode = parallel_mode
        #  args
        self.in_units = margin_params.get("feature_dim", 512)
        self.out_units = margin_params["class_num"]
        self.start_label = margin_params.get("start_label", 0)
        self.end_label = margin_params.get("end_label", 0)
        #  lambda parameter
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
        self.weight = Parameter(torch.Tensor(self.out_units, self.in_units))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, label):
        #  weight normalize
        w = self.weight
        w_norm = F.normalize(w, dim=1)
        # w.data[:] = w_norm.data
        #  cos_theta = x'w/|x|
        x_norm = x.norm(p=2, dim=1) #  [batch, 1]
        cos_theta = x.mm(w_norm.t()) #  [batch, out]
        cos_theta = cos_theta / x_norm.view(-1, 1)
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
        if self.parallel_mode == "DataParallel":
            index.scatter_(1, label.data.view(-1, 1), 1)
            index = index.byte()
        elif self.parallel_mode == "ModelParallel":
            for i in range(label.size()[0]):
                label_val = int(label.data[i])
                if self.start_label <= label_val < self.end_label:
                    index[i][label_val-self.start_label] = 1
        index = Variable(index)
        #  output
        lamb = self.__get_lambda()
        output = x_norm_cos_theta * 1.0  # size=(B,Classnum)
        if self.parallel_mode == "DataParallel":
            output[index] -= x_norm_cos_theta[index] / (1 + lamb)
            output[index] += x_norm_phi_theta[index] / (1 + lamb)
            theta_mean = theta[index].mean()
        elif self.parallel_mode == "ModelParallel":
            output = output - x_norm_cos_theta * index / (1 + lamb)
            output = output + x_norm_phi_theta * index / (1 + lamb)
            theta_mean = 0.0
        #  write tensorboard summary
        self.__write_summary(lamb, theta_mean)
        return output

    def __get_lambda(self):
        val = self.lamb_base * (1.0 + self.lamb_gamma * 
                                self.common_dict["global_step"]) ** (-self.lamb_power)
        val = max(self.lamb_min, val)
        if self.common_dict["global_step"] % 500 == 0 and torch.cuda.current_device() == 0:
            print ("Now lambda = {}".format(val))
        return val

    def __write_summary(self, lamb, theta_mean):
        current_index = torch.cuda.current_device()
        if self.common_dict["global_step"] % 10 == 0 and self.common_dict["tensorboard_writer"]:
            self.common_dict["tensorboard_writer"].add_scalar("lambda_{}".format(current_index),
                                                              lamb,
                                                              self.common_dict["global_step"])
            self.common_dict["tensorboard_writer"].add_scalar("theta_mean_{}".format(current_index),
                                                              theta_mean,
                                                              self.common_dict["global_step"])


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
    def __init__(self, gpu_num, model_params={}, parallel_mode="DataParallel", common_dict={}):
        super(SphereFaceNet, self).__init__()
        self.feature_dim = model_params.get("feature_dim", 512)
        self.class_num = model_params["class_num"]
        self.parallel_mode = parallel_mode

        layer_type = model_params.get("layer_type", "20layer")
        image_size = model_params.get("image_size", 112)
        if self.parallel_mode == "DataParallel":
            self.main_net = self.__main_net(layer_type, self.feature_dim, image_size)
            self.margin_fc = MarginInnerProduct(model_params, parallel_mode, common_dict)
            self.ce_loss = nn.CrossEntropyLoss(reduce=False)
        elif self.parallel_mode == "ModelParallel":
            self.main_net = nn.DataParallel(self.__main_net(layer_type, self.feature_dim, image_size))
            self.main_net.cuda()

            #  split chunk number same as gpu number
            self.num_chunk_margin_fc = gpu_num
            self.margin_fc_chunks = nn.ModuleList()
            start_index = 0
            for i in range(self.num_chunk_margin_fc):
                _class_num = self.class_num / self.num_chunk_margin_fc
                if self.class_num % self.num_chunk_margin_fc > i:
                    _class_num += 1
                _model_params = copy.deepcopy(model_params)
                _model_params["class_num"] = _class_num
                _model_params["start_label"] = start_index
                _model_params["end_label"] = start_index + _class_num
                self.margin_fc_chunks.append(MarginInnerProduct(_model_params,
                                             parallel_mode, common_dict).cuda(i))
                start_index += _class_num

            self.ce_loss = nn.DataParallel(nn.CrossEntropyLoss(reduce=False))
            self.ce_loss.cuda()

    def forward(self, x, label):
        x = self.main_net(x)
        if self.parallel_mode == "DataParallel":
            x = self.margin_fc(x, label)
        elif self.parallel_mode == "ModelParallel":
            x_list = []
            for i in range(self.num_chunk_margin_fc):
                _x = self.margin_fc_chunks[i](x.cuda(i), label.cuda(i))
                x_list.append(_x.cuda(0))
            x = torch.cat(x_list, dim=1)
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
              "lamb_gamma": 0.0001, "lamb_power": 1, "lamb_min": 10,
              "class_num": 6, "feature_dim": 4}
    test = MarginInnerProduct(params)

    x = Variable(torch.ones(2, 4).float())
    # label = Variable(torch.ones(2).long())
    label = Variable(torch.FloatTensor([1, 5]))
    print test(x, label)
