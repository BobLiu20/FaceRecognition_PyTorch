import sys

from sphere_face_net import SphereFaceNet

# init
def init(model, **kwargs):
    func = getattr(sys.modules["models"], model)
    return func(**kwargs)

if __name__ == "__main__":
    import models
    net = models.init("SphereFaceNet", feature_dim=512, label_num=6, model_params={})

