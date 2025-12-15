import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from models.arcface import ArcFaceHead

class PlayerFusionEmbeddingModel(nn.Module):
    """
    CNN backbone -> image embedding
    Tabular MLP   -> tabular embedding
    concat -> fuse -> ArcFace

    forward(img, tab, labels=None):
      labels!=None: ArcFace logits (for CE)
      labels==None: normalized fused embedding
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        num_tabular_features: int,
        embedding_dim: int = 512,
        pretrained: bool = True,
        arcface_s: float = 30.0,
        arcface_m: float = 0.5,
        tabular_hidden: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
        )
        backbone_out = self.backbone.num_features

        self.img_head = nn.Sequential(
            nn.BatchNorm1d(backbone_out),
            nn.Dropout(dropout),
            nn.Linear(backbone_out, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

        self.use_tabular = num_tabular_features > 0
        self.num_tabular_features = num_tabular_features

        if self.use_tabular:
            self.tab_head = nn.Sequential(
                nn.BatchNorm1d(num_tabular_features),
                nn.Linear(num_tabular_features, tabular_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(tabular_hidden, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
            self.fuse = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim, bias=False),
                nn.BatchNorm1d(embedding_dim),
            )
        else:
            self.tab_head = None
            self.fuse = None

        self.arcface = ArcFaceHead(
            in_features=embedding_dim,
            out_features=num_classes,
            s=arcface_s,
            m=arcface_m,
        )

        self.embedding_dim = embedding_dim

    def get_embedding(self, img: torch.Tensor, tab: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(img)
        img_emb = self.img_head(feat)

        if self.use_tabular:
            assert tab is not None, "tabular features required but tab is None"
            tab_emb = self.tab_head(tab)
            fused = self.fuse(torch.cat([img_emb, tab_emb], dim=1))
        else:
            fused = img_emb

        return F.normalize(fused, p=2, dim=1)

    def forward(self, img: torch.Tensor, tab: torch.Tensor | None = None, labels: torch.Tensor | None = None) -> torch.Tensor:
        feat = self.backbone(img)
        img_emb = self.img_head(feat)

        if self.use_tabular:
            tab_emb = self.tab_head(tab)
            fused = self.fuse(torch.cat([img_emb, tab_emb], dim=1))
        else:
            fused = img_emb

        if labels is not None:
            return self.arcface(fused, labels)
        return F.normalize(fused, p=2, dim=1)