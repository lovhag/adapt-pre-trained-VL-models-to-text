import torch
import multiprocessing
import json
import h5py
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import List
from PIL import Image
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader


CLIP_EMBED_DIM = 512


def main(args):
    model = VisualOnlyCLIPModel()
    model = DataParallel(model, device_ids=args.device_ids) if torch.cuda.is_available() else model

    images_and_ids = [json.loads(line) for line in args.input]
    image_files = [ex["path"] for ex in images_and_ids]
    image_ids = [ex["image_id"].encode("utf-8") for ex in images_and_ids]
    image_ds = ImageDataset(image_files)
    image_dl = DataLoader(image_ds, batch_size=args.batch_size, num_workers=multiprocessing.cpu_count())

    feature_buffer = h5py.File(args.output, "w")
    feature_ds = feature_buffer.create_dataset("features", shape=(len(image_ds), CLIP_EMBED_DIM), dtype=np.float32)
    ids_ds = feature_buffer.create_dataset("ids", shape=(len(image_ds), ), dtype=f"S{max(len(id) for id in image_ids)}")
    ids_ds[...] = np.array(image_ids, dtype="S")

    with torch.no_grad():
        for indices, batch in tqdm(image_dl):
            batch = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in batch.items()}
            clip_features = model(**batch)
            feature_ds[indices.tolist(), :] = clip_features.cpu()

    feature_buffer.flush()
    feature_buffer.close()


class VisualOnlyCLIPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().to("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, *args, **kwargs):
        return self.model.get_image_features(*args, **kwargs)


class ImageDataset(Dataset):
    def __init__(self, image_files: List[str]):
        super().__init__()
        self.image_files = image_files
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.processor.feature_extractor = PatchedCLIPFeatureExtractor("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        pil_image = Image.open(self.image_files[index])
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        example = self.processor(images=pil_image, return_tensors="pt")
        example["pixel_values"] = example["pixel_values"].squeeze()
        return index, example


# TODO: To fix bug in resizing
from transformers import CLIPFeatureExtractor
import PIL
class PatchedCLIPFeatureExtractor(CLIPFeatureExtractor):
    def resize(self, image, size, resample=PIL.Image.BILINEAR):
        """
        Resizes :obj:`image`. Note that this will trigger a conversion of :obj:`image` to a PIL Image.
        Args:
            image (:obj:`PIL.Image.Image` or :obj:`np.ndarray` or :obj:`torch.Tensor`):
                The image to resize.
            size (:obj:`int` or :obj:`Tuple[int, int]`):
                The size to use for resizing the image.
            resample (:obj:`int`, `optional`, defaults to :obj:`PIL.Image.BILINEAR`):
                The filter to user for resampling.
        """
        self._ensure_format_supported(image)

        if not isinstance(size, tuple):
            scale_factor = max(*image.size) / min(*image.size)
            size = (int(scale_factor * size), size) if image.size[0] > image.size[1] else (size, int(scale_factor * size))
        if not isinstance(image, PIL.Image.Image):
            image = self.to_pil_image(image)

        return image.resize(size, resample=resample)


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=sys.stdin, type=argparse.FileType('r', encoding="utf-8"), help="Json lines with \"image_id\" and \"path\"")
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--device-ids", default=None, help="Comma separated list of device ids to use")
    parser.add_argument("--output", required=True, help="Output file (*.hdf5)")
    args = parser.parse_args()

    if args.device_ids is not None:
        args.device_ids = [int(device) for device in args.device_ids.split(",")]

    main(args)
