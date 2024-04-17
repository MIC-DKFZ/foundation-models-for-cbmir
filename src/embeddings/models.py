from pathlib import Path
import torch
from torch import nn
from PIL import Image
from medclip import MedCLIPModel, MedCLIPVisionModelViT, constants
from open_clip import create_model_from_pretrained, create_model_and_transforms

from transformers import (
    AutoImageProcessor,
    ViTModel,
    ViTMAEModel,
    CLIPFeatureExtractor,
    SamModel,
    SamImageProcessor,
    AutoModel,
    Dinov2Model,
    ResNetForImageClassification,
)


class BaseModel(nn.Module):
    """Base class for all models to reduce redundancy in model initialization."""

    def __init__(self, model_name, checkpoint_path: Path, device: str = None):
        super().__init__()
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = self.load_model(model_name, checkpoint_path)
        self.processor = self.load_processor(model_name, checkpoint_path)
        self.model.to(self.device).eval()

    def load_model(self, model_name, checkpoint_path):
        """Method to load model, can be overridden based on specific model requirements."""
        return AutoModel.from_pretrained(model_name, cache_dir=checkpoint_path)

    def load_processor(self, model_name, checkpoint_path):
        """Method to load processor, can be overridden based on specific processor requirements."""
        return AutoImageProcessor.from_pretrained(model_name, cache_dir=checkpoint_path)

    def process_image(self, img):
        """Process an image using the loaded processor."""
        return self.processor(img)

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        """Method to get image embeddings, should be overridden by subclasses."""
        raise NotImplementedError("This method should be overridden by subclasses.")


class SAM(BaseModel):
    """Specific model class for SAM using base class features."""

    def load_model(self, model_name, checkpoint_path):
        return SamModel.from_pretrained(model_name, cache_dir=checkpoint_path)

    def load_processor(self, model_name, checkpoint_path):
        return SamImageProcessor.from_pretrained(model_name, cache_dir=checkpoint_path)

    def process_image(self, img):
        return self.processor(img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.get_image_embeddings(inputs.to(self.device))
        return embeddings.mean(axis=[2, 3]).cpu()


class SAMViTB(SAM):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "facebook/sam-vit-base", checkpoint_path / self.__class__.__name__, device
        )


class MedSAMViTB(SAM):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "wanglab/medsam-vit-base", checkpoint_path / self.__class__.__name__, device
        )

    def download_model(self, checkpoint_path):
        import requests

        url = "https://drive.usercontent.google.com/download?id=1UAmWL88roYR7wKlnApw5Bcuzf2iQgk6_&export=download&authuser=0&confirm=t&uuid=18b8846f-ab8f-4043-864d-79968ca56b8e&at=APZUnTWSRMCs0A19GEAldZeRj4Zm:1715085525148"

        response = requests.get(url)
        with open(checkpoint_path, "wb") as f:
            f.write(response.content)

    def load_model(self, _, checkpoint_path):
        from segment_anything import sam_model_registry

        # The Vision encoder in wanglab/medsam-vit-base is not correct.
        # We first download the model from the original source and then load the model
        full_checkpoint_path = checkpoint_path / "medsam_vit_b.pth"
        if not full_checkpoint_path.exists():
            full_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.download_model(full_checkpoint_path)

        return sam_model_registry["vit_b"](
            checkpoint=full_checkpoint_path,
        )

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.image_encoder(inputs.to(self.device))
        return embeddings.mean(axis=[2, 3]).cpu()


class ViT(BaseModel):
    """Specific model class for Vision Transformer using base class features."""

    def load_model(self, model_name, checkpoint_path):
        return ViTModel.from_pretrained(model_name, cache_dir=checkpoint_path)

    def process_image(self, img):
        return self.processor(img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(inputs.to(self.device)).last_hidden_state[:, 0, :].cpu()
        return embeddings


class ViTB16(ViT):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "google/vit-base-patch16-224-in21k",
            checkpoint_path=checkpoint_path / self.__class__.__name__,
            device=device,
        )


class ViTMAEB(ViT):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "facebook/vit-mae-base",
            checkpoint_path=checkpoint_path / self.__class__.__name__,
            device=device,
        )

    def load_model(self, model_name, checkpoint_path):
        return ViTMAEModel.from_pretrained(model_name, cache_dir=checkpoint_path)


class DINOv2(ViT):
    def load_model(self, model_name, checkpoint_path):
        return Dinov2Model.from_pretrained(model_name, cache_dir=checkpoint_path)


class DINOv2ViTB14(DINOv2):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "facebook/dinov2-base", checkpoint_path / self.__class__.__name__, device
        )


class BiomedCLIPVitB16_224(BaseModel):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            checkpoint_path / self.__class__.__name__,
            device,
        )

    def load_model(self, model_name, checkpoint_path):
        model, _ = create_model_from_pretrained(model_name, cache_dir=checkpoint_path)
        return model

    def load_processor(self, model_name, checkpoint_path):
        _, processor = create_model_from_pretrained(
            model_name, cache_dir=checkpoint_path
        )
        return processor

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode_image(inputs.to(self.device)).cpu()
        return embeddings


class MedCLIP(BaseModel):
    """Specific model class for MedCLIP using base class features."""

    def __init__(self, checkpoint_path: Path, device: str = None):
        super().__init__(None, checkpoint_path / self.__class__.__name__, device)

    def load_model(self, model_name, checkpoint_path):
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained(str(checkpoint_path))
        return model

    def load_processor(self, model_name, checkpoint_path):
        return CLIPFeatureExtractor(
            do_resize=True,
            size=224,
            resample=Image.BICUBIC,
            do_center_crop=True,
            crop_size=224,
            do_normalize=True,
            image_mean=constants.IMG_MEAN,
            image_std=constants.IMG_STD,
            do_convert_rgb=False,
            do_pad_square=True,
        )

    def process_image(self, img):
        return self.processor(img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode_image(inputs.to(self.device)).cpu()
        return embeddings


class ResNet(BaseModel):
    """Specific model class for ResNet using base class features."""

    def load_model(self, model_name, checkpoint_path):
        return ResNetForImageClassification.from_pretrained(
            model_name, cache_dir=checkpoint_path
        )

    def process_image(self, img):
        return self.processor(img, return_tensors="pt")["pixel_values"][0]

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model(inputs.to(self.device)).logits.cpu()
        return embeddings


class ResNet50(ResNet):
    def __init__(self, checkpoint_path: Path, device: str = None):
        super().__init__(
            "microsoft/resnet-50", checkpoint_path / self.__class__.__name__, device
        )


class CLIP(BaseModel):
    """Specific model class for CLIP using base class features."""

    def __init__(
        self,
        model_name,
        pretrained,
        checkpoint_path: Path,
        device: str = None,
    ):
        super().__init__(
            f"{model_name}//{pretrained}",
            checkpoint_path,
            device,
        )

    def load_model(self, model_name, checkpoint_path):
        model_str, pretrained = model_name.split("//")
        model, _, _ = create_model_and_transforms(
            model_str,
            pretrained=pretrained,
            cache_dir=checkpoint_path,
        )
        return model

    def load_processor(self, model_name, checkpoint_path):
        model_str, pretrained = model_name.split("//")
        _, _, processor = create_model_and_transforms(
            model_str,
            pretrained=pretrained,
            cache_dir=checkpoint_path,
        )
        return processor

    @torch.no_grad()
    def get_image_embeddings(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encode_image(inputs.to(self.device)).cpu()
        return embeddings


class CLIPOpenAIViTL14(CLIP):
    def __init__(self, checkpoint_path: Path, device: str = None) -> None:
        super().__init__(
            model_name="ViT-L-14",
            pretrained="openai",
            checkpoint_path=checkpoint_path / self.__class__.__name__,
            device=device,
        )
