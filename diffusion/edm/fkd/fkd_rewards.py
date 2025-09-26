import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import hpsv2
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import timm

# from image_reward_utils import rm_load
# from llm_grading import LLMGrader

# Stores the reward models
REWARDS_DICT = {
    "Clip-Score": None,
    "ImageReward": None,
    "LLMGrader": None,
    "Diversity": None,
}


_CLS_MODEL: Optional[nn.Module] = None
_CLS_DEVICE: Optional[torch.device] = None
_CLS_TRANSFORM: Optional[Callable] = None
_LABEL_TO_IDX: Optional[Dict[str, int]] = None
_NUM_CLASSES: Optional[int] = None


def setup_classifier_reward(
    create_model_fn: Callable[[], nn.Module],
    checkpoint_path: str,
    *,
    device: Optional[Union[str, torch.device]] = None,
    num_classes: Optional[int] = None,
    input_size: int = 256,
    crop_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),  # ImageNet defaults
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    half: bool = False,
    label_to_idx: Optional[Dict[str, int]] = None,
) -> None:
    """
    create_model_fn must return an initialized model with the correct head (num_classes).
    """
    global _CLS_MODEL, _CLS_DEVICE, _CLS_TRANSFORM, _LABEL_TO_IDX, _NUM_CLASSES
    _CLS_DEVICE = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _CLS_MODEL = create_model_fn().to(_CLS_DEVICE)
    _NUM_CLASSES = num_classes

    state = torch.load(checkpoint_path, map_location="cpu")
    # accept either raw state_dict or {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    _CLS_MODEL.load_state_dict(state, strict=True)
    _CLS_MODEL.eval()
    if half:
        _CLS_MODEL.half()

    _CLS_TRANSFORM = T.Compose(
        [
            T.Resize(input_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_size),
            T.ToTensor(),  # [0,1]
            T.Normalize(mean=mean, std=std),
        ]
    )
    _LABEL_TO_IDX = label_to_idx


def _images_to_batch(images: list[Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Accepts list of PIL or CHW float tensors in [0,1].
    Returns a batch tensor (B,3,H,W) normalized for the classifier.
    """
    assert len(images) > 0, "images is empty."
    pil_list = []
    for im in images:
        if isinstance(im, Image.Image):
            pil_list.append(im)
        elif torch.is_tensor(im):
            # tensor image: convert to PIL through [0,1] range
            x = im.detach().float()
            if x.dim() == 3 and x.shape[0] in (1, 3):  # (C,H,W)
                x = x
            elif x.dim() == 3 and x.shape[-1] in (1, 3):  # (H,W,C)
                x = x.permute(2, 0, 1)
            else:
                raise ValueError("Tensor image must be CHW or HWC with C in {1,3}.")
            # if model outputs in [-1,1], remap to [0,1]
            if x.min() < 0:
                x = (x.clamp(-1, 1) * 0.5) + 0.5
            x = x.clamp(0, 1)
            pil_list.append(T.ToPILImage()(x))
        else:
            raise TypeError("images must be a list of PIL.Image or torch.Tensor.")
    batch = torch.stack([_CLS_TRANSFORM(im) for im in pil_list], dim=0)
    return batch


# Returns the reward function based on the guidance_reward_fn name
def get_reward_function(reward_name, images, prompts, metric_to_chase="overall_score", **kwargs):
    # if reward_name != "LLMGrader":
    #     print("`metric_to_chase` will be ignored as it only applies to 'LLMGrader' as the `reward_name`")
    # if reward_name == "ImageReward":
    #     return do_image_reward(images=images, prompts=prompts)

    if reward_name == "Clip-Score":
        return do_clip_score(images=images, prompts=prompts)

    elif reward_name == "HumanPreference":
        return do_human_preference_score(images=images, prompts=prompts)

    # elif reward_name == "LLMGrader":
    #     return do_llm_grading(images=images, prompts=prompts, metric_to_chase=metric_to_chase)

    elif reward_name == "ClassifierLossReward":
        return do_classifier_loss_score(images=images, prompts=prompts, **kwargs)

    elif reward_name == "PixelReward":
        return do_pixel_score(images=images, prompts=prompts, **kwargs)

    elif reward_name == "Diversity":
        return do_diversity_score(images=images, feature_pool=kwargs.pop("feature_pool"), **kwargs)

    elif reward_name == "AFT":
        aft_module = kwargs.pop("aft_module")
        score = kwargs.pop("score")
        targets = kwargs.pop("targets", None)
        feature_pool_model = kwargs.pop("feature_pool_model", None)
        feature_pool_pretrained = kwargs.pop("feature_pool_pretrained", None)
        return do_aft_score(images=images, aft_module=aft_module, score=score, targets=targets, feature_pool_model=feature_pool_model, feature_pool_pretrained=feature_pool_pretrained)

    else:
        raise ValueError(f"Unknown metric: {reward_name}")


def do_classifier_loss_score(*, images, prompts, model, mode="prob", temperature=1.0):

    B = len(images)
    device = model.device

    batch = _images_to_batch(images).to(device)
    labels = prompts

    logits = model(batch) / max(1e-6, float(temperature))
    probs = logits.softmax(dim=-1)

    # probability for the provided (target) label
    p_t = probs.gather(1, labels.view(-1, 1)).squeeze(1)

    if mode == "binary":
        preds = probs.argmax(dim=-1)
        rewards = (preds != labels).float()
    elif mode == "prob":
        rewards = 1.0 - p_t  # smooth [0,1]
    elif mode == "margin":
        top2_p, top2_i = probs.topk(2, dim=1)
        # probability of best "other" class:
        p_other = torch.where(top2_i[:, 0] == labels, top2_p[:, 1], top2_p[:, 0])
        rewards = (p_other - p_t).clamp_min(0)  # >=0 if others outrank the label
    else:
        raise ValueError("mode must be one of: 'binary', 'prob', 'nll', 'margin'.")

    return rewards.detach().float().cpu().tolist()


def do_pixel_score(*, images, prompts):
    # to check FK steering works correctly with EDM
    def _to_pil_from_tensor_batch(x):
        """
        x: (B, C, H, W), image-space float tensor.
        EDM typically uses [-1, 1] range. Adjust if your model uses [0,1].
        """
        x = x.detach().float().clamp(-1, 1)
        x = (x * 0.5) + 0.5  # [-1,1] -> [0,1]
        x = (x * 255).clamp(0, 255).to(torch.uint8)
        x = x.permute(0, 2, 3, 1).cpu().numpy()  # (B,H,W,C)
        return [Image.fromarray(arr) for arr in x]

    images = _to_pil_from_tensor_batch(images) if torch.is_tensor(images) else images

    rewards = []
    for img in images:
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        s = np.uint64(arr).sum()
        rewards.append(float(s))

    arr0 = np.array(images[0])
    H, W = arr0.shape[:2]
    C = 1 if arr0.ndim == 2 else arr0.shape[2]
    denom = max(1.0, H * W * C)
    rewards = [r / denom for r in rewards]  # -> [0,1]

    return rewards


# Compute human preference score
def do_human_preference_score(*, images, prompts, use_paths=False):
    if use_paths:
        scores = hpsv2.score(images, prompts, hps_version="v2.1")
        scores = [float(score) for score in scores]
    else:
        scores = []
        for i, image in enumerate(images):
            score = hpsv2.score(image, prompts[i], hps_version="v2.1")
            # print(f"Human preference score for image {i}: {score}")
            score = float(score[0])
            scores.append(score)

    # print(f"Human preference scores: {scores}")
    return scores


# Compute CLIP-Score and diversity
def do_clip_score_diversity(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        arr_clip_result = []
        arr_img_features = []
        for i, prompt in enumerate(prompts):
            clip_result, feature_vect = REWARDS_DICT["Clip-Score"].score(
                prompt, images[i], return_feature=True
            )

            arr_clip_result.append(clip_result.item())
            arr_img_features.append(feature_vect["image"])

    # calculate diversity by computing pairwise similarity between image features
    diversity = torch.zeros(len(images), len(images))
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            diversity[i, j] = (arr_img_features[i] - arr_img_features[j]).pow(2).sum()
            diversity[j, i] = diversity[i, j]
    n_samples = len(images)
    diversity = diversity.sum() / (n_samples * (n_samples - 1))

    return arr_clip_result, diversity.item()


# # Compute ImageReward
# def do_image_reward(*, images, prompts):
#     global REWARDS_DICT
#     if REWARDS_DICT["ImageReward"] is None:
#         REWARDS_DICT["ImageReward"] = rm_load("ImageReward-v1.0")

#     with torch.no_grad():
#         image_reward_result = REWARDS_DICT["ImageReward"].score_batched(prompts, images)
#         # image_reward_result = [REWARDS_DICT["ImageReward"].score(prompt, images[i]) for i, prompt in enumerate(prompts)]

#     return image_reward_result


# Compute CLIP-Score
def do_clip_score(*, images, prompts):
    global REWARDS_DICT
    if REWARDS_DICT["Clip-Score"] is None:
        REWARDS_DICT["Clip-Score"] = CLIPScore(download_root=".", device="cuda")
    with torch.no_grad():
        clip_result = [
            REWARDS_DICT["Clip-Score"].score(prompt, images[i]) for i, prompt in enumerate(prompts)
        ]
    return clip_result


# # Compute LLM-grading
# def do_llm_grading(*, images, prompts, metric_to_chase="overall_score"):
#     global REWARDS_DICT

#     if REWARDS_DICT["LLMGrader"] is None:
#         REWARDS_DICT["LLMGrader"] = LLMGrader()
#     llm_grading_result = [
#         REWARDS_DICT["LLMGrader"].score(
#             images=images[i], prompts=prompt, metric_to_chase=metric_to_chase
#         )
#         for i, prompt in enumerate(prompts)
#     ]
#     return llm_grading_result


"""
@File       :   CLIPScore.py
@Time       :   2023/02/12 13:14:00
@Auther     :   Jiazheng Xu
@Contact    :   xjz22@mails.tsinghua.edu.cn
@Description:   CLIPScore.
* Based on CLIP code base
* https://github.com/openai/CLIP
"""


class CLIPScore(nn.Module):
    def __init__(self, download_root, device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(
            "ViT-L/14", device=self.device, jit=False, download_root=download_root
        )

        if device == "cpu":
            self.clip_model.float()
        else:
            clip.model.convert_weights(
                self.clip_model
            )  # Actually this line is unnecessary since clip by default already on float16

        # have clip.logit_scale require no grad.
        self.clip_model.logit_scale.requires_grad_(False)

    def score(self, prompt, pil_image, return_feature=False):
        # if (type(image_path).__name__=='list'):
        #     _, rewards = self.inference_rank(prompt, image_path)
        #     return rewards

        # text encode
        text = clip.tokenize(prompt, truncate=True).to(self.device)
        txt_features = F.normalize(self.clip_model.encode_text(text))

        # image encode
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = F.normalize(self.clip_model.encode_image(image))

        # score
        rewards = torch.sum(torch.mul(txt_features, image_features), dim=1, keepdim=True)

        if return_feature:
            return rewards, {"image": image_features, "txt": txt_features}

        return rewards.detach().cpu().numpy().item()
    

class TimmWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.classifier = model.get_classifier()
        self.model.reset_classifier(0)
        self.feature_reshape = lambda x: x
        if isinstance(self.classifier, nn.Linear):
            self.feat_dim = self.classifier.in_features
        elif isinstance(self.classifier, nn.Conv2d): # resnetv2
            self.feat_dim = self.classifier.in_channels
            # reshape a flattened feature into 1x1 image
            self.feature_reshape = lambda x: x.reshape(x.shape[0], self.feat_dim, 1, 1)
        else:
            self.feat_dim = None
    
    def forward(self, x, return_feat=False):
        if isinstance(x, dict):
            x = x['image']
        feat = self.model(x)
        feat = self.feature_reshape(feat) # resnetv2
        out = self.classifier(feat)
        out = out.reshape(feat.shape[0], -1) # resnetv2
        feat = feat.reshape(feat.shape[0], -1) # resnetv2
        if return_feat:
            return out, feat
        return out

class DiversityModel(nn.Module):
    def __init__(self,
        model_class,
        num_features,
        diag,
        tensor_product,
        scale_ckpt,
        **kwargs
    ):
        super().__init__()
        # model, get_transform, tokenizer, input_collate_fn = models.create_model(model_class, out_dim=0, pretrained=True, extract_features=True, **kwargs)
        model = timm.create_model(model_class, num_classes=0, pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        get_transform = lambda train: timm.data.create_transform(**data_config, is_training=train)
        self.transform = get_transform(train=False)
        model = TimmWrapper(model)
        self.model = model.cuda().eval()

        self.diag = diag
        self.tensor_product = tensor_product

        if self.diag:
            self.s = nn.Parameter(torch.zeros(num_features)).to("cuda")
        else:
            self.s = nn.Parameter(torch.randn(num_features, num_features) / (num_features ** 0.5)).to("cuda")
        scale_state = torch.load(scale_ckpt, map_location='cpu')
        self.s.data.copy_(scale_state['s'])

    def get_scaled_feature(self, x):
        x = self.transform(x)
        feat = self.model(x).detach()
        if self.tensor_product:
            # Split features according to feat_dims
            split_features = torch.split(feat, self.feat_dims, dim=1)
            # Split scales according to feat_dims
            split_scales = torch.split(self.s.sigmoid(), self.feat_dims, dim=0)
            # Scale each axis
            scaled_features = [f * s for f, s in zip(split_features, split_scales)]
            # Compute tensor product
            scaled_feat = scaled_features[0]
            for i in range(1, len(scaled_features)):
                scaled_feat = torch.einsum('bi,bj->bij', scaled_feat, scaled_features[i]).reshape(scaled_feat.shape[0], -1)
        else:
            if self.diag:
                scaled_feat = feat * self.s.sigmoid()
            else:
                scaled_feat = feat @ self.s # (B, d) @ (d, d) -> (B, d)

        return scaled_feat
    
    def score(self, images, feature_pool):
        features = self.get_scaled_feature(images)

        diversity_scores = []
        for feat in features:
            # feat_cand = torch.cat([feature_pool, feat.unsqueeze(0)], dim=0)
            # calculate channelwise std of feat_cand
            # std_feat_cand = torch.std(feat_cand, dim=0)
            # diversity_score = std_feat_cand.mean().item()
            # diversity_scores.append(diversity_score)
            
            # # calculate pairwise distance between feature_pool and feat using cosine similarity
            cos_sim = F.cosine_similarity(feature_pool, feat.unsqueeze(0), dim=1)
            if len(cos_sim) == 0:
                diversity_score = 0.0
            else:
                diversity_score = (1 - cos_sim).min().item() # higher is more diverse
            diversity_scores.append(diversity_score)

            # calculate pairwise distance between feature_pool and feat
            # dists = torch.norm(feature_pool - feat.unsqueeze(0), dim=1, p=2) # (N,)
            # # diversity_score = dists.mean().item() # higher is more diverse
            # if len(dists) == 0:
            #     diversity_score = 0.0
            # else:
            #     diversity_score = dists.min().item() # higher is more diverse
            # diversity_scores.append(diversity_score)

        return diversity_scores
        

def do_diversity_score(*, images, feature_pool, **kwargs):
    global REWARDS_DICT
    if REWARDS_DICT["Diversity"] is None:
        REWARDS_DICT["Diversity"] = DiversityModel(**kwargs)

    with torch.no_grad():
        diversity_result = REWARDS_DICT["Diversity"].score(images, feature_pool)
    return diversity_result


def do_aft_score(*, images, aft_module, score, targets=None, feature_pool_model=None, feature_pool_pretrained=None, **kwargs):
    if score == "ce":
        rewards = aft_module.get_ce_loss(images, targets, reduction="none").detach().cpu().tolist()
    elif score == "aft":
        rewards = []
        for image in images:
            reward = aft_module.get_aft_loss(image.unsqueeze(0), feature_pool_model, feature_pool_pretrained).detach().cpu().item()
            rewards.append(reward)
    elif score == "total":
        rewards = []
        for image, target in zip(images, targets):
            reward = aft_module.get_total_loss(image.unsqueeze(0), target.unsqueeze(0), feature_pool_model, feature_pool_pretrained).detach().cpu().item()
            rewards.append(reward)
    else:
        raise ValueError(f"Unknown score: {score}")
    return rewards