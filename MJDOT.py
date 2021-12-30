import os
import argparse
import random
import scipy
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import ot
import data
from scipy.spatial.distance import cdist


parser = argparse.ArgumentParser()
parser.add_argument('--IMG_DIR', default='DATAMISS/office31_10_3/', help='dataset location')#
parser.add_argument('--IMG_NAME', default='AD', help='source_target')#
parser.add_argument('--feature_norm', type=str, default='none', choices=['none', 'l2', 'unit-gaussian'])
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--seed', type=int, default=137)
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--n_iter', type=int, default=1000)

args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

FN = torch.from_numpy
dil = data.index_labels
join = os.path.join


# **************************************** load dataset ****************************************

dset = data.XianDataset(args.IMG_DIR, args.IMG_NAME, feature_norm=args.feature_norm)
_Xs = FN(dset.feature_s).to(args.device)
_Ys = FN(dil(dset.label_s, dset.Call)).to(args.device)  # indexed labels
_Xt = FN(dset.feature_t).to(args.device)
_Yt = FN(dil(dset.label_t, dset.Call)).to(args.device)
class_num = np.size(dset.Call)

# **************************************** create data loaders ****************************************
sampling_weights = None
xy_s_iter = data.Iterator(
    [_Xs, _Ys],
    args.batch_size,
    shuffle=True,
    sampling_weights=sampling_weights,
    continuous=False)
xy_t_iter = data.Iterator(
    [_Xt, _Yt],
    args.batch_size,
    shuffle=True,
    sampling_weights=sampling_weights,
    continuous=False)
label_iter = data.Iterator([torch.arange(dset.n_Call, device=args.device)], args.batch_size)
class_iter = data.Iterator([torch.arange(dset.n_Call)], 1)


# **************************************** define network ****************************************

class FF(nn.Module):
    def __init__(self, num_features):
        super(FF, self).__init__()
        self.bottleneck_layer1 = nn.Linear(num_features, 512)
        self.bottleneck_layer1.apply(init_weights)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer1, nn.ReLU(), nn.Dropout(0.05))#
        self.classifier_layer = nn.Linear(512, class_num)
        self.classifier_layer.apply(init_weights)
        self.predict_layer = nn.Sequential(self.bottleneck_layer, self.classifier_layer)

    def forward(self, x):
        out = self.bottleneck_layer(x)
        outC = self.classifier_layer(out)
        return (out, outC)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def DIST(x, y=None):
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


def loss_hinge(Y,F):
    res = np.zeros((Y.shape[0], F.shape[0]))
    for i in range(Y.shape[1]):
        res += np.maximum(0, 1 - Y[:, i].reshape((Y.shape[0], 1)) * F[:, i].reshape((1, F.shape[0]))) ** 2
    return res


def test(data, label, model):
    with torch.no_grad():
        output = model(data)
        _, predict = torch.max(output, 1)
        accuracy = torch.sum(torch.squeeze(predict) == label).item() / float(label.size()[0])
    return accuracy


# **************************************** train classifier ****************************************

num_features = 2048
net = FF(num_features)
net = net.to(args.device)
net.train(True)

criterion = nn.CrossEntropyLoss()
optimizer_dict = [{"params": filter(lambda p: p.requires_grad, net.bottleneck_layer.parameters()), "lr": 0.01},
                  {"params": filter(lambda p: p.requires_grad, net.classifier_layer.parameters()), "lr": 0.01}]
optimizer = optim.SGD(optimizer_dict, lr=0.1, momentum=0.9, weight_decay=0.005, nesterov=True)

test_max = 0

x_r = torch.zeros(args.batch_size, num_features).cuda()
y_r = torch.zeros(args.batch_size, 1).squeeze().cuda()
x_m = torch.zeros(args.batch_size, num_features).cuda()
y_m = torch.zeros(args.batch_size, 1).squeeze().cuda()
alpha1 = 50


num_iter = args.n_iter
for iter_num in range(1, num_iter + 1):
    print(iter_num)
    net.train(True)
    optimizer.zero_grad()

    x_s, y_s = next(xy_s_iter)
    x_t, y_t = next(xy_t_iter)

    outS, outSL = net(x_s)
    outT, outTL = net(x_t)

    #Mixup data
    lam = np.random.beta(alpha1, alpha1)
    # print('lam', lam)
    x_m = lam * x_s + (1 - lam) * x_r
    outM, outML = net(x_m)
    outML = outML.narrow(0, 0, len(y_s.squeeze()))
    loss_m = lam * criterion(outML, y_s.long()) + (1 - lam) * criterion(outML, y_r.long())
    x_r = x_s
    y_r = y_s

    _, s_m, _ = torch.svd(outM)
    _, s_t, _ = torch.svd(outT)
    sdist = DIST(s_m.reshape(-1, 1), s_t.reshape(-1, 1))

    reg = 1e-1
    gamma = 1
    alpha = 100

    C0 = cdist(outM.cpu().detach().numpy(), outT.cpu().detach().numpy(), metric='sqeuclidean')
    OUTTL = outTL.narrow(0, 0, len(y_t.squeeze()))
    C11 = loss_hinge(y_s.reshape(-1, 1).cpu().detach().numpy(), outTL.cpu().detach().numpy())
    C12 = loss_hinge(y_r.reshape(-1, 1).cpu().detach().numpy(), outTL.cpu().detach().numpy())
    C1 = lam * C11 + (1 - lam) * C12
    C = alpha * C0 + C1
    OUTM = ot.unif(outM.shape[0])
    OUTT = ot.unif(outT.shape[0])
    gamma = ot.emd(OUTM, OUTT, C)
    gamma = np.float32(gamma)
    gamma = FN(gamma).to(args.device)
    gdist = DIST(outM, outT)
    ldist = DIST(outML, outTL)
    JDOT_loss_m = 0.001 * torch.sum(gamma * (gdist + ldist + sdist))


    #SVD loss
    _, s_s, _ = torch.svd(outS)
    _, s_t, _ = torch.svd(outT)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    sigma_loss = 0.0001 * sigma
    sdist = DIST(s_s.reshape(-1,1), s_t.reshape(-1,1))



    #CLASSIFIER loss
    OUTSL = outSL.narrow(0, 0, len(y_s.squeeze()))
    classif_loss = criterion(OUTSL, y_s.squeeze())


    total_loss = 0.01*(JDOT_loss_m) + 10*loss_m #+ 1*classif_loss + 


    total_loss.backward()
    optimizer.step()

    if (iter_num % 100) == 0:
        net.eval()


    test_acc = test(_Xt, _Yt, net.predict_layer)
    print('test_acc:%.4f' % (test_acc))

    if test_acc > test_max:
        test_max = test_acc

print('test_max:%.4f' % (test_max))

with torch.no_grad():
    outS, outSL = net(_Xs)
    outT, outTL = net(_Xt)
    f_s = outS.cpu().numpy()
    f_t = outT.cpu().numpy()

np.savetxt('DW_source.csv', f_s , delimiter = ',')
np.savetxt('DW_target.csv', f_t, delimiter = ',')
