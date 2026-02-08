import torch
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small


class RAFTFlowExtractor:
    def __init__(self, device: torch.device, max_flow: float = 20.0) -> None:
        weights = Raft_Small_Weights.DEFAULT
        self.transforms = weights.transforms()
        self.model = raft_small(weights=weights, progress=True).to(device).eval()
        self.device = device
        self.max_flow = max_flow

    @torch.no_grad()
    def compute(self, clip: torch.Tensor) -> torch.Tensor:
        flows = []
        for i in range(clip.shape[0] - 1):
            img1 = clip[i].unsqueeze(0).to(self.device)
            img2 = clip[i + 1].unsqueeze(0).to(self.device)
            img1, img2 = self.transforms(img1, img2)
            pred = self.model(img1, img2)[-1][0].cpu()
            pred = torch.clamp(pred / self.max_flow, -1.0, 1.0)
            flows.append(pred)
        return torch.stack(flows, dim=0)
