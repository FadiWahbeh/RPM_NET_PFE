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
    def __init__(self, num_iterations=5, use_dsc_lite=True, dsc_k=12):
        super().__init__()
        self.num_iterations = num_iterations
        self.emb_dims = 512
        self.encoder = DGCNN_Embedding(emb_dims=self.emb_dims)

        self.use_dsc_lite = use_dsc_lite
        self.dsc_k = dsc_k

    @staticmethod
    def _knn_idx(x, k):
        # x: (B,N,3)
        with torch.no_grad():
            xx = (x ** 2).sum(dim=2, keepdim=True)
            dist = xx - 2 * (x @ x.transpose(1, 2)) + xx.transpose(1, 2)
            idx = dist.topk(k=k, dim=-1, largest=False)[1]
        return idx

    def _dsc_lite_reweight(self, src_xyz, tgt_xyz, P):
        y = torch.matmul(P, tgt_xyz)          # (B,N,3)
        k = min(self.dsc_k, src_xyz.shape[1] - 1)
        if k <= 1:
            return P

        idx = self._knn_idx(src_xyz, k)        # (B,N,k)
        B, N, _ = src_xyz.shape

        # ✅ FIX OOM: indexation plate O(B·N·k) au lieu de O(B·N²)
        # L'ancienne version faisait .expand(B, N, N, 3) ce qui allouait
        # un tenseur de B×N×N×3 floats (~192MB pour B=8, N=1024) → OOM GPU.
        batch_offset = torch.arange(B, device=src_xyz.device).view(B, 1, 1) * N
        idx_flat = (idx + batch_offset).reshape(-1)          # (B·N·k,)

        src_flat = src_xyz.reshape(B * N, 3)
        y_flat   = y.reshape(B * N, 3)

        src_nb = src_flat[idx_flat].view(B, N, k, 3)         # (B,N,k,3)
        y_nb   = y_flat[idx_flat].view(B, N, k, 3)           # (B,N,k,3)

        d_src = torch.norm(src_nb - src_xyz.unsqueeze(2), dim=3)   # (B,N,k)
        d_y   = torch.norm(y_nb   - y.unsqueeze(2),       dim=3)   # (B,N,k)

        med = torch.median(d_src.reshape(B, -1), dim=1).values \
                  .view(B, 1, 1).clamp(min=1e-6)
        score = torch.exp(-torch.abs(d_src - d_y) / med)
        conf  = score.mean(dim=2, keepdim=True)               # (B,N,1)

        P2 = P * conf
        P2 = P2 / (P2.sum(dim=2, keepdim=True) + 1e-8)
        return P2

    def compute_weighted_procrustes(self, src, tgt, weights, eps=1e-8):
        """
        src: (B,N,3)
        tgt: (B,N,3)
        weights: (B,N,N)
        AMP-safe: SVD done in float32.
        """
        w_row = weights.sum(dim=2, keepdim=True) + eps
        tgt_corr = torch.matmul(weights, tgt) / w_row

        src_centroid = (src * w_row).sum(dim=1, keepdim=True) / (w_row.sum(dim=1, keepdim=True) + eps)
        tgt_centroid = (tgt_corr * w_row).sum(dim=1, keepdim=True) / (w_row.sum(dim=1, keepdim=True) + eps)

        src_centered = src - src_centroid
        tgt_centered = tgt_corr - tgt_centroid

        H = torch.matmul((src_centered * w_row).transpose(1, 2), tgt_centered)  # (B,3,3)

        # ✅ AMP fix: SVD in float32
        H32 = H.float()

        try:
            U, S, Vh = torch.linalg.svd(H32, full_matrices=False)
            V = Vh.transpose(-2, -1)
        except Exception:
            U, S, V = torch.svd(H32)

        R = torch.matmul(V, U.transpose(1, 2))

        det = torch.det(R)
        diag_fix = torch.ones((src.size(0), 3), device=src.device, dtype=torch.float32)
        # ✅ FIX sign(0): torch.sign(0) = 0 détruit la 3e colonne de R si det ≈ 0
        # torch.where garantit -1 ou +1, jamais 0.
        diag_fix[:, 2] = torch.where(det < 0,
                             torch.full_like(det, -1.0),
                             torch.ones_like(det))
        R = torch.matmul(torch.matmul(V, torch.diag_embed(diag_fix)), U.transpose(1, 2))

        R = R.to(dtype=src.dtype)

        t = tgt_centroid.squeeze(1) - torch.matmul(R, src_centroid.squeeze(1).unsqueeze(2)).squeeze(2)
        return R, t

    def forward(self, src, tgt, return_timings=False):
        # src,tgt: (B,3,N)
        src_p = src.transpose(1, 2)  # (B,N,3)
        tgt_p = tgt.transpose(1, 2)  # (B,N,3)

        B = src.size(0)
        R_acc = torch.eye(3, device=src.device, dtype=src.dtype).view(1, 3, 3).repeat(B, 1, 1)
        t_acc = torch.zeros(B, 3, device=src.device, dtype=src.dtype)

        src_curr = src_p.clone()
        transforms = []
        iter_times = []

        feat_tgt = self.encoder(tgt).transpose(1, 2)  # (B,N,Emb)

        for _ in range(self.num_iterations):
            if return_timings and src.is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            feat_src = self.encoder(src_curr.transpose(1, 2)).transpose(1, 2)  # (B,N,Emb)

            scores = torch.matmul(feat_src, feat_tgt.transpose(1, 2))  # (B,N,N)
            P = sinkhorn(scores)

            if self.use_dsc_lite:
                P = self._dsc_lite_reweight(src_curr, tgt_p, P)

            dR, dt = self.compute_weighted_procrustes(src_curr, tgt_p, P)

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
