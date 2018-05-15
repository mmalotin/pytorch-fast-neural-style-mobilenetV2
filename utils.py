import torch
import torch.nn.functional as F
from PIL import Image


def load_im(f, size=None):
    img = Image.open(f)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    return img


def save_im(f, tens):
    img = tens.detach().clamp(0, 255).cpu().numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(f)


def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b, c, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c*h*w)
    return gram


def norm_batch(b):
    mean = b.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = b.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    b = b.div_(255.0)
    return (b - mean) / std


def regularization_loss(x):
    loss = (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))
    return loss


def prepare_target_wass(tens):
    b, c, h, w = tens.size()
    t = tens.view(b, c, h * w)
    means = t.mean(keepdim=True, dim=2)
    t = t - means
    tt = t.transpose(1, 2)
    covar = t.bmm(tt)/(h*w)
    values = []
    vects = []
    traces = []
    for i in range(covar.size(0)):
        vals, vecs = covar[i, :, :].eig(True)
        vals_mat = F.relu(vals[:, 0]).sqrt().diag()
        values.append(vals_mat)
        vects.append(vecs)
        traces.append(F.relu(vals[:, 0]).sum())
    values = torch.stack(values)
    vecs = torch.stack(vects)
    traces = torch.stack(traces)
    covar_root = vecs.bmm(values).bmm(vecs.transpose(1, 2))
    return means, traces, covar_root


def prepare_preds_wass(tens):
    b, c, h, w = tens.size()
    t = tens.view(b, c, h * w)
    means = t.mean(keepdim=True, dim=2)
    t = t - means
    tt = t.transpose(1, 2)
    covar = t.bmm(tt)/(h*w)
    tr = [x.trace() for x in torch.functional.unbind(covar)]
    tr = torch.stack(tr)
    return means, tr, covar


def wasserstein_loss(preds, target):
    mean_target, trace_target, root_cov_target = target
    mean_pred, trace_pred, cov_pred = prepare_preds_wass(preds)
    mean_diff = (mean_target - mean_pred)
    mean_diff2 = mean_diff.mul(mean_diff).sum(1, True).squeeze()
    covar_prod = root_cov_target.bmm(cov_pred).bmm(root_cov_target)
    tr2 = [x.eig()[0][:, 0] for x in torch.functional.unbind(covar_prod)]
    tr2 = torch.stack(tr2)
    var_overlap = F.relu(tr2).sqrt().sum(1, True).squeeze()
    return (mean_diff2 + trace_target + trace_pred - 2*var_overlap).sum()
