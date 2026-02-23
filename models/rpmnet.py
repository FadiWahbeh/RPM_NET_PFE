import time
import torch
import torch.nn as nn

try:
    from models.dgcnn import DGCNN_Embedding
except Exception:
    from dgcnn import DGCNN_Embedding

try:
    from utils.transform import sinkhorn
except Exception:
    from transform import sinkhorn


class RPMNet(nn.Module):
    def __init__(self, num_iterations=5):
        super(RPMNet, self).__init__()
        self.num_iterations = num_iterations
        self.emb_dims = 512
        self.encoder = DGCNN_Embedding(emb_dims=self.emb_dims)

        self.weights_net = nn.Sequential(
            nn.Linear(self.emb_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def compute_weighted_procrustes(self, src, tgt, weights, eps=1e-8):
        """
        src: (B,N,3)
        tgt: (B,N,3)
        weights: (B,N,N)
        """
        w_row = weights.sum(dim=2, keepdim=True) + eps
        tgt_corr = torch.matmul(weights, tgt) / w_row

        src_centroid = (src * w_row).sum(dim=1, keepdim=True) / (w_row.sum(dim=1, keepdim=True) + eps)
        tgt_centroid = (tgt_corr * w_row).sum(dim=1, keepdim=True) / (w_row.sum(dim=1, keepdim=True) + eps)

        src_centered = src - src_centroid
        tgt_centered = tgt_corr - tgt_centroid

        H = torch.matmul((src_centered * w_row).transpose(1, 2), tgt_centered)

        U, S, V = torch.svd(H)
        R = torch.matmul(V, U.transpose(1, 2))

        det = torch.det(R)
        diag = torch.ones((src.size(0), 3), device=src.device)
        diag[:, 2] = torch.sign(det)
        R = torch.matmul(torch.matmul(V, torch.diag_embed(diag)), U.transpose(1, 2))

        t = tgt_centroid.squeeze(1) - torch.matmul(R, src_centroid.squeeze(1).unsqueeze(2)).squeeze(2)
        return R, t

    def forward(self, src, tgt, return_timings: bool = False):
        # src,tgt: (B,3,N)
        src_p = src.transpose(1, 2)  # (B,N,3)
        tgt_p = tgt.transpose(1, 2)  # (B,N,3)

        B = src.size(0)
        R_acc = torch.eye(3, device=src.device).view(1, 3, 3).repeat(B, 1, 1)
        t_acc = torch.zeros(B, 3, device=src.device)

        src_curr = src_p.clone()
        transforms = []
        iter_times = []

        feat_tgt = self.encoder(tgt)           # (B,Emb,N)
        feat_tgt = feat_tgt.transpose(1, 2)    # (B,N,Emb)

        for _ in range(self.num_iterations):
            if return_timings and src.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            feat_src = self.encoder(src_curr.transpose(1, 2))
            feat_src = feat_src.transpose(1, 2)  # (B,N,Emb)

            scores = torch.matmul(feat_src, feat_tgt.transpose(1, 2))  # (B,N,N)
            match_matrix = sinkhorn(scores)

            dR, dt = self.compute_weighted_procrustes(src_curr, tgt_p, match_matrix)

            src_curr = torch.matmul(src_curr, dR.transpose(1, 2)) + dt.unsqueeze(1)

            R_acc = torch.matmul(dR, R_acc)
            t_acc = torch.matmul(dR, t_acc.unsqueeze(2)).squeeze(2) + dt

            transforms.append((R_acc, t_acc))

            if return_timings and src.is_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if return_timings:
                iter_times.append(t1 - t0)

        if return_timings:
            return R_acc, t_acc, transforms, iter_times
        return R_acc, t_acc, transforms