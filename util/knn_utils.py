import torch


# Get point clouds KNN index
def point_knn(points, num_neighbor=10):
    """
    :param points: tensor with shape of BxNxC
    :param num_neighbor: int number to select nearest neighbors
    :return:index: tensor with shape of BxNxnum_neighbor
    """
    inner = -2*torch.matmul(points, points.transpose(2, 1))
    xx = torch.sum(points**2, dim=2, keepdim=True)
    pairwise_distance = -xx-inner-xx.transpose(2, 1)
    index = pairwise_distance.topk(k=num_neighbor, dim=-1)[1]

    return index


# Get neighbor feature based on the neighbor index
def get_neighbor_feature(features, num_neighbor, idx=None, device=torch.device('cuda:0')):
    """
    :param features: combined feature and coord with the shape of BxNX(3+C)
    :param num_neighbor: int number to select nearest neighbors
    :param idx: neighbor index with the shape of BxNxnum_neighbor
    :param device: device where the calculations are taken
    :return: features: combined feature and coord with the shape of BxNxnum_neighborx(3+C)
    """
    batch_size, num_point, feature_dim = features.shape
    if idx is None:
        idx = point_knn(features[:, :, 0:3], num_neighbor)
    
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_point
    idx = idx + idx_base
    idx = idx.view(-1)
    features = features.transpose(2, 1).contiguous()
    features = features.view(batch_size*num_point, -1)[idx, :]
    features = features.view(batch_size, num_point, num_neighbor, feature_dim)
    
    return features


if __name__ == '__main__':
    
    a = torch.randn((8, 10, 6))

    # idx = point_knn(a, 5)

    b = get_neighbor_feature(a, 5, None, device=torch.device('cpu'))

    print(b.size())