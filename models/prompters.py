import torch
import torch.nn as nn
from torchvision import transforms

import kornia
from kornia.geometry.transform.imgwarp import get_affine_matrix2d


def safe_logit(x, eps=1e-7):
    x = torch.clamp(x, min=eps, max=1-eps)
    return torch.log(x / (1 - x))


class ACAVP(nn.Module):
    def __init__(self, args):
        super(ACAVP, self).__init__()
        self.sigma_range = args.sigma_range
        self.image_size = args.image_size
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # initialize multiplicative_prompt
        self._multiplicative_prompt = nn.Parameter(
        safe_logit(torch.ones([1, 3, self.image_size, self.image_size])/self.sigma_range),
            )
        
        # initialize affine_prompt
        scale_init = torch.tensor(args.scale_init).repeat(1, 2)
        scale_init = safe_logit(scale_init)
        self._t = torch.nn.Parameter(torch.zeros(2))
        self._theta = torch.nn.Parameter(torch.zeros(1))
        self._s = torch.nn.Parameter(scale_init)
        self._sh = torch.nn.Parameter(torch.zeros(2))
        self.affine_mode = getattr(args, 'mode', 'bilinear')
        self.affine_padding_mode = getattr(args, 'padding_mode', 'zeros')
        self.affine_align_corners = getattr(args, 'align_corners', True)
        self.affine_center = getattr(args, 'affine_center', True)
        self.theta_range = getattr(args, 'theta_range', 0.5)
        self.sh_range = getattr(args, 'sh_range', 0.5)
        self.t_range = getattr(args, 't_range', 0.25)
        self.center = torch.tensor([0, 0]).float()
        # pad prompt
        self.prompt = torch.nn.Parameter(
            torch.zeros((3, self.image_size, self.image_size)).float(), 
            requires_grad=True)
        self.mask = None

        if not args.use_clip_model:
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            self.denormalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225],
            )
        else:
            self.normalize = transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
                )
            self.denormalize = transforms.Normalize(
                                    mean=[-0.48145466/0.26862954, 
                                          -0.4578275/0.26130258, 
                                          -0.40821073/0.27577711],
                                    std=[1/0.26862954, 
                                         1/0.26130258, 
                                         1/0.27577711]
                                    )

    def limit_transform(self):
        # rotate
        theta = self.tanh(self._theta) * self.theta_range
        angle = theta * (180 / torch.pi)
        # scale
        s = self.sigmoid(self._s)
        # shear
        sh = self.tanh(self._sh) * self.sh_range
        # translate
        t = self.tanh(self._t)*self.image_size * self.t_range
        return angle, s, sh, t

    def forward(self, x):
        # denrom 
        x = self.denormalize(x) # => 0 ~ 1

        # Affine VP
        angle, s, sh, t = self.limit_transform()
        self.angle = angle.expand(x.shape[0])
        self.s = s.expand(x.shape[0], 2)
        self.sh = sh.expand(x.shape[0], -1)
        self.t = t.expand(x.shape[0], -1)
        center = self.center.expand(x.shape[0], -1).to(x.device)
        affine_matrix = get_affine_matrix2d(
            self.t, 
            center, 
            self.s, 
            -self.angle, 
            sx=self.sh[..., 0], 
            sy=self.sh[..., 1],
            )
        x = kornia.geometry.transform.affine(x, 
                   affine_matrix[..., :2, :3], 
                   self.affine_mode, 
                   self.affine_padding_mode, 
                   self.affine_align_corners,
                   )   

        # Multiplicative VP
        self.multiplicative_prompt = self._multiplicative_prompt.repeat(x.size(0), 1, 1, 1)
        multiplicative_prompt = self.sigmoid(self.multiplicative_prompt) * self.sigma_range
        x = x * multiplicative_prompt    

        # Padding VP
        self.mask = torch.all(x==0, dim=(0,1)).long()
        self.padding_prompt = self.prompt.repeat(x.size(0), 1, 1, 1)
        padding_prompt = self.mask.to(x.device) * self.padding_prompt.to(x.device)
        x = x + padding_prompt
        x = self.normalize(x)

        return x 