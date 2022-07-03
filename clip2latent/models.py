import clip
import torch.nn.functional as F
from torchvision import transforms
import torch

def load_sg(network_pkl):
    import sys
    sys.path.append("stylegan3")
    import dnnlib
    import legacy

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    return G


class Clipper(torch.nn.Module):
    def __init__(self, clip_variant):
        super().__init__()
        clip_model, _ = clip.load(clip_variant, device="cpu")
        self.clip = clip_model
        self.normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        self.clip_size = (224,224)

    def embed_image(self, image):
        """Expects images in -1 to 1 range"""
        clip_in = F.interpolate(image, self.clip_size, mode="area")
        clip_in = self.normalize(0.5*clip_in + 0.5).clamp(0,1)
        return self.clip.encode_image(self.normalize(clip_in))

    def embed_text(self, text_samples):
        text = clip.tokenize(text_samples).to(self._get_device())
        return self.clip.encode_text(text)

    def _get_device(self):
        for p in self.clip.parameters():
            return p.device