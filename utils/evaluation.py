#!/usr/bin/python
# _____________________________________________________________________________
import time

# ----------------
# import libraries
# ----------------

# standard libraries
# -----
import torch
from torch.linalg import lstsq
import torch.nn.functional as F
import pacmap
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm



@torch.no_grad()
def knn_evaluation(args, train_features, train_labels, test_features, test_labels, n_classes):
    i=0
    correct20=0
    k=20
    size_batch = args.knn_batch_size
    sim_func = lambda x, x_pair: F.cosine_similarity(x.unsqueeze(1), x_pair.unsqueeze(0), dim=2)

    expanded_train_label = train_labels.view(1,-1).expand(size_batch,-1)
    retrieval_one_hot = torch.zeros((size_batch*20, n_classes), device=train_features.device)

    while i < test_features.shape[0]:
        endi = min(i+size_batch,test_features.shape[0])
        tf = test_features[i:endi]
        distance_matrix = sim_func(tf, train_features)
        valk, indk = torch.topk(distance_matrix, k, dim=1)
        if tf.shape[0] < size_batch:
            retrieval_one_hot = torch.zeros(tf.shape[0] * k, n_classes, device=distance_matrix.device)
            expanded_train_label = train_labels.view(1, -1).expand(tf.shape[0], -1)

        retrieval = torch.gather(expanded_train_label, 1, indk)
        rt_onehot = retrieval_one_hot.scatter(1, retrieval.view(-1, 1), 1)
        rt_onehot = rt_onehot.view(retrieval.shape[0], k, n_classes)
        not_available = (rt_onehot.sum(dim=1) == 0)
        sim_topk = rt_onehot * valk.unsqueeze(-1)

        probs = torch.sum(sim_topk, dim=1)
        probs[not_available] = -10000
        prediction = torch.max(probs, dim=1).indices
        correct20 += (prediction == test_labels[i:endi]).sum(dim=0)

        i = endi

    return correct20/test_features.shape[0]



# custom functions
# -----
@torch.no_grad()
def get_representations(args, net, data_loader, t, get_pair=False):
    """
    Get all representations of the dataset given the network and the data loader
    params:
        args: arguments
        net: the network to be used (torch.nn.Module)
        data_loader: data loader of the dataset (DataLoader)
    return:
        tuple of data with the first one being image representations. Other data depends on the dataset.
    """
    net.eval()
    gathered_data = [[], [], [], [], [], [], [], [], [], []]
    strt_idx = 0 if not get_pair else 1

    for data in tqdm(data_loader):
        gathered_data[0].append(net(t(data[0][0])))
        if get_pair:
            gathered_data[1].append(net(t(data[0][1])))
        for i in range(1, len(data)):
            gathered_data[i+strt_idx].append(data[i])
        if args.name == "test3":
            break
    tensor_data = [torch.cat(data, dim=0) for data in gathered_data if data]
    return tensor_data


@torch.no_grad()
def lls_fit(train_features, train_labels, n_classes):
    """
        Fit a linear least square model
        params:
            train_features: the representations to be trained on (Tensor)
            train_labels: labels of the original data (LongTensor)
            n_classes: int, number of classes
        return:
            ls: the trained lstsq model (torch.linalg) 
    """
    ls = lstsq(train_features, F.one_hot(train_labels, n_classes).type(torch.float32))
    
    return ls

@torch.no_grad()
def lls_eval(trained_lstsq_model, eval_features, eval_labels):
    """
    Evaluate a trained linear least square model
    params:
        trained_lstsq_model: the trained lstsq model (torch.linalg)
        eval_features: the representations to be evaluated on (Tensor)
        eval_labels: labels of the data (LongTensor)
    return:
        acc: the LLS accuracy (float)
    """
    acc = ((eval_features @ trained_lstsq_model.solution).argmax(dim=-1) == eval_labels).sum() / len(eval_features)
    return acc

@torch.no_grad()
def wcss_bcss(representations, labels, n_classes):
    """
        Calculate the within-class and between-class average distance ratio
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
        return:
            wb: the within-class and between-class average distance ratio (float)
    """
    # edgecase: there might be less on one class than of another
    # -----
    # representations = torch.stack([representations[labels == i] for i in range(n_classes)])
    # centroids = representations.mean(1, keepdim=True)
    # wcss = (representations - centroids).norm(dim=-1).mean()
    # bcss = F.pdist(centroids.squeeze()).mean()
    # wb = wcss / bcss
    representations = [representations[labels == i] for i in range(n_classes)]
    centroids = torch.stack([r.mean(0, keepdim=True) for r in representations])
    wcss = [(r - centroids[i]).norm(dim=-1) for i,r in enumerate(representations)]
    wcss = torch.cat(wcss).mean()
    bcss = F.pdist(centroids.squeeze()).mean()
    wb = wcss / bcss
    return wb


class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self):
            self.reset()
        
        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
        
        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the
    specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def supervised_eval(model, dataloader, criterion, no_classes):
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    
    for idx, (images, labels) in enumerate(dataloader):
        images = images.float()
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        bsz = labels.shape[0]
        
        # forward
        _, output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        
        top1.update(acc1[0], bsz)
        
    print('Test: [{0}/{1}]\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
              idx, len(dataloader),
              loss=losses, top1=top1))
        
    return top1.avg, losses.avg
    

@torch.no_grad()
def get_pacmap(args, representations, labels, epoch, n_classes, class_labels):
    """
        Draw the PacMAP plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
        return:
            fig: the PacMAP plot (matplotlib.figure.Figure)
    """
    # sns.set()
    sns.set_style("ticks")
    sns.set_context('paper', font_scale=1.8, rc={'lines.linewidth': 2})
    # color_map = get_cmap('viridis')
    color_map = ListedColormap(sns.color_palette('colorblind', 50))
    #legend_patches = [Patch(color=color_map(i / n_classes), label=label) for i, label in enumerate(class_labels)]
    legend_patches = [Patch(color=color_map(i), label=label) for i, label in enumerate(class_labels)]
    # save the visualization result
    embedding = pacmap.PaCMAP(2)
    X_transformed = embedding.fit_transform(representations.cpu().numpy(), init="pca")
    fig, ax = plt.subplots(1, 1, figsize=(7.7,4.8))
    labels = labels.cpu().numpy()
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels, cmap=color_map, s=0.6)
    ax.set_title(args.main_loss + r' $N_{fix}$=' + str(args.n_fix))
    plt.xticks([]), plt.yticks([])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1.), handles=legend_patches, fontsize=13.8)
    # ax.text(-0.15, 1.05, 'C', transform=ax.transAxes, size=30, weight='medium')
    plt.xlabel(f'Epoch: {epoch}')
    return fig


def cosine_similarity(p_vec, q_vec):
    """
    cosine_similarity takes two numpy arrays of the same shape and returns
    a float representing the cosine similarity between two vectors
    """
    p_vec, q_vec = p_vec.flatten(), q_vec.flatten()
    return np.dot(p_vec, q_vec) / (np.linalg.norm(p_vec) * np.linalg.norm(q_vec))


@torch.no_grad()
def get_neighbor_similarity(representations, labels, epoch, sim_func=cosine_similarity):
    """
        Draw a similarity plot
        params:
            representations: the representations to be evaluated (Tensor)
            labels: labels of the original data (LongTensor)
            epoch: epoch (int)
            sim_func: similarity function with two parameters
        return:
            fig: similarity plot (matplotlib.figure.Figure)
    """

    unique_labels = torch.unique(labels)

    if len(labels) != len(unique_labels):
        # calculate the mean over representations
        rep_centroid = torch.zeros([len(unique_labels), representations.shape[-1]])
        for i in range(len(unique_labels)):
            rep_centroid[i] = representations[torch.where(labels == i)[0]].mean(0)

        list_of_indices = np.arange(len(unique_labels))
        labels = list_of_indices
        representations = rep_centroid
        n_samples_per_object = 1

    else:
        list_of_indices = np.arange(len(labels))
        n_samples_per_object = 1

    distances = np.zeros([len(unique_labels), len(unique_labels)])

    # Fill a distance matrix that relates every representation of the batch
    for i in list_of_indices:
        for j in list_of_indices:
            distances[labels[i], labels[j]] += sim_func(representations[i].cpu(), representations[j].cpu())
            # distances[labels[i], labels[j]] += 1

    distances /= n_samples_per_object ** 2  # get the mean distances between representations

    # get some basic statistics
    # print('[INFO:] distance', distances.max(), distances.min(), distances.std())

    # duplicate the matrix such that you don't get to the edges when
    # gathering distances
    distances = np.hstack([distances, distances, distances])
    # plt.matshow(distances)
    # plt.show()

    # how many neighbors do you want to show (n_neighbors = n_classes for sanity check, you would have to see a global symmetry)
    n_neighbors = len(unique_labels)
    topk_dist_plus = np.zeros([len(labels), n_neighbors])
    topk_dist_minus = np.zeros([len(labels), n_neighbors])

    for k in range(n_neighbors):
        for i in range(len(unique_labels)):
            topk_dist_plus[i, k] += distances[i, i + len(unique_labels) + k]
            topk_dist_minus[i, k] += distances[i, i + len(unique_labels) - k]

    topk_dist = np.vstack([topk_dist_plus, topk_dist_minus])

    fig, ax = plt.subplots()
    ax.errorbar(np.arange(0, n_neighbors), topk_dist.mean(0), marker='.', markersize=10, xerr=None,
                yerr=topk_dist.std(0))
    ax.set_title('representation similarity')
    ax.set_xlabel('nth neighbour')
    ax.set_ylabel('cosine similarity')
    ax.set_ylim(-1.1, 1.1)
    ax.hlines(topk_dist.mean(0)[n_neighbors // 2:].mean(), -100, 100, color='gray', linestyle='--')
    ax.set_xlim(-2, n_neighbors + 2)

    return fig

# _____________________________________________________________________________

# Stick to 80 characters per line
# Use PEP8 Style
# Comment your code

# -----------------
# top-level comment
# -----------------

# medium level comment
# -----

# low level comment
