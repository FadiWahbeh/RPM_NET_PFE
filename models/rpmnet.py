import torch
import torch.nn as nn
from models.dgcnn import DGCNN_Embedding
from utils.transform import sinkhorn

class RPMNet(nn.Module):
    def __init__(self, num_iterations=5):
        super(RPMNet, self).__init__()
        self.num_iterations = num_iterations
        self.emb_dims = 512
        
        # 1. Feature Extractor
        self.encoder = DGCNN_Embedding(emb_dims=self.emb_dims)
        
        # 2. Parameter Prediction
        self.weights_net = nn.Sequential(
            nn.Linear(self.emb_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) 
        )

    # SVD Pondéré (Weighted SVD)
    def compute_weighted_procrustes(self, src, tgt, weights):
        # src, tgt: (B, N, 3)
        # weights: (B, N, N)
        weights_sum = torch.sum(weights, dim=2, keepdim=True) + 1e-8
        
        # Barycentres pondérés
        src_mean = torch.matmul(weights, src) / weights_sum
        tgt_mean = tgt.clone()
        
        # On doit recentrer les points pour SVD
        # Pour faire simple dans cette implémentation, on utilise SVD classique sur les correspondances
        # RPM : Y_hat = Weights * Target
        tgt_corr = torch.matmul(weights, tgt)
        
        src_centered = src - src.mean(dim=1, keepdim=True)
        tgt_corr_centered = tgt_corr - tgt_corr.mean(dim=1, keepdim=True)
        
        # Matrice de Covariance H
        H = torch.matmul(src_centered.transpose(1, 2), tgt_corr_centered)
        
        U, S, V = torch.svd(H)
        
        R = torch.matmul(V, U.transpose(1, 2))
        
        # Determinant
        det = torch.det(R)
        diag = torch.ones((src.size(0), 3), device=src.device)
        diag[:, 2] = torch.sign(det)
        R = torch.matmul(torch.matmul(V, torch.diag_embed(diag)), U.transpose(1, 2))
        
        t = tgt_corr.mean(dim=1, keepdim=True).transpose(1,2) - torch.matmul(R, src.mean(dim=1, keepdim=True).transpose(1,2))
        
        return R, t.squeeze(2)

    def forward(self, src, tgt):
        # src, tgt : (B, 3, N)
        src_p = src.transpose(1, 2)
        tgt_p = tgt.transpose(1, 2)
        
        batch_size = src.size(0)
        
        # Initialisation Transformation (Identité)
        R_acc = torch.eye(3).view(1, 3, 3).repeat(batch_size, 1, 1).to(src.device)
        t_acc = torch.zeros(batch_size, 3).to(src.device)
        
        src_curr = src_p.clone()
        transforms = []
        
        # Extraction Features Cible 
        feat_tgt = self.encoder(tgt) # (B, Emb, N)
        feat_tgt = feat_tgt.transpose(1, 2) # (B, N, Emb)
        
        for i in range(self.num_iterations):
            # 1. Extraction Features Source
            # On doit remettre en (B, 3, N) pour DGCNN
            feat_src = self.encoder(src_curr.transpose(1, 2)) 
            feat_src = feat_src.transpose(1, 2) # (B, N, Emb)
            
            # 2. Similarity Score
            # (B, N, Emb) @ (B, Emb, N) -> (B, N, N)
            scores = torch.matmul(feat_src, feat_tgt.transpose(1, 2))
            
            # 3. Sinkhorn (Soft Assign)
            match_matrix = sinkhorn(scores)
            
            # 4. SVD Pondéré
            dR, dt = self.compute_weighted_procrustes(src_curr, tgt_p, match_matrix)
            
            # 5. Update Source
            # (B, N, 3) @ (B, 3, 3)^T = (B, N, 3)
            src_curr = torch.matmul(src_curr, dR.transpose(1, 2)) + dt.unsqueeze(1)
            
            # 6. Accumulate
            R_acc = torch.matmul(dR, R_acc)
            t_acc = torch.matmul(dR, t_acc.unsqueeze(2)).squeeze(2) + dt
            
            transforms.append((R_acc, t_acc))
            
        return R_acc, t_acc, transforms