import argparse
import importlib
import time
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from retrieval import load_index


def main(args):
    # Load model
    model_module = importlib.import_module("embeddings.models")
    model = getattr(model_module, args.model)(checkpoint_path=args.checkpoint_path)

    # Load FAISS index and metadata
    index_path = args.index_dir / "faiss.index"
    metadata_path = args.index_dir / "metadata.json"
    index, metadata = load_index(index_path, metadata_path)

    if args.img_dir:
        img_paths = (
            list(args.img_dir.rglob("*.png"))
            + list(args.img_dir.rglob("*.jpg"))
            + list(args.img_dir.rglob("*.jpeg"))
        )
    else:
        img_paths = [args.img_path]

    results = []

    for img_path in tqdm(img_paths, desc="Processing images", unit="images"):
        img = Image.open(img_path).convert("RGB")
        processed_img = model.process_image(img)

        if processed_img.dim() == 3:
            processed_img = processed_img.unsqueeze(0)

        features = model.get_image_embeddings(processed_img)

        _, indices = index.search(features, args.n)

        result_dict = {"query_path": img_path}
        for i, idx in enumerate(indices[0]):
            result_dict[f"similar_{i+1}"] = metadata[idx]

        results.append(result_dict)

    results_df = pd.DataFrame(results)
    output_dir = args.output_dir / time.strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_dir / "results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--img_path",
        type=Path,
        help="Path to specific img to evalute",
        default=None,
    )
    parser.add_argument(
        "--img_dir",
        type=Path,
        help="Path to directory containing images",
        default=None,
    )
    parser.add_argument(
        "--output_dir", type=Path, help="Path to save the results", default="outputs"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        help="Path where the model checkpoints of the pretrained models are stored",
        default="checkpoints",
    )
    parser.add_argument(
        "--index_dir",
        type=Path,
        help="Path where the index and metadata of faiss are stored",
        default="experiments/chexpert_mimic_nih14_radimagenet/BiomedCLIPVitB16_224/",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Class name of the model to use for embeddings creation",
        default="BiomedCLIPVitB16_224",
    )
    parser.add_argument(
        "--n",
        type=int,
        help="Number of most similar images to retrieve",
        default=10,
    )
    parser.add_argument("--batch_size", type=int, help="Batch size", default=16)

    args = parser.parse_args()
    assert args.img_path or args.img_dir, "Either img_path or img_dir must be provided"
    main(args)
