import os
import sys
import json
import argparse
from pathlib import Path
from PIL import Image

import cv2
import torch
import torchvision
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

# InsightFace for on-the-fly bbox detection
from insightface.app import FaceAnalysis

# Add facexformer to path
sys.path.append(str(Path("./facexformer").resolve()))
from network import FaceXFormer

# Standard FairFace mappings used by FaceXFormer
AGE_MAP = {0: "0-2", 1: "3-9", 2: "10-19", 3: "20-29", 4: "30-39", 5: "40-49", 6: "50-59", 7: "60-69", 8: "70+"}
GENDER_MAP = {0: "Male", 1: "Female"}
RACE_MAP = {0: "White", 1: "Black", 2: "Latino_Hispanic", 3: "East Asian", 4: "Southeast Asian", 5: "Indian", 6: "Middle Eastern"}

def get_video_path(corpus, clip_id, share_dir):
    share = Path(share_dir)
    if corpus == "voxceleb2_test":
        parts = clip_id.split("__")
        return share / "vox-celeb-dataset/vox-celeb-raw/mp4" / parts[0] / parts[1] / f"{parts[2]}.mp4"
    elif corpus == "hdtf":
        stem = clip_id.split("__", 1)[1]
        return share / "HDTF-dataset/HDTF-RGB" / f"{stem}.mp4"
    elif corpus == "mead":
        parts = clip_id.split("____")
        return share / "MEAD/videos" / parts[0] / "video" / parts[1] / parts[2] / parts[3] / f"{parts[4]}.mp4"
    else:
        raise ValueError(f"Unsupported corpus: {corpus}")

def main(args):
    device = f"cuda:{args.gpu_num}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading InsightFace (SCRFD)...")
    app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=args.gpu_num, det_size=(640, 640))

    print("Loading FaceXFormer...")
    model = FaceXFormer().to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict_backbone'])
    model.eval()

    transforms_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(224, 224), interpolation=InterpolationMode.BICUBIC),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    workdir = Path(args.workdir)
    anno_dir = workdir / "annotations" / args.corpus
    
    face_detect_path = anno_dir / "face_detect.json"
    demo_path = anno_dir / "demographics.json"

    with open(face_detect_path, "r") as f:
        face_data = json.load(f)

    demo_results = {}
    if demo_path.exists():
        with open(demo_path, "r") as f:
            demo_results = json.load(f)
        print(f"Resuming {args.corpus}: Loaded {len(demo_results)} existing annotations.")

    clips_to_process = [cid for cid in face_data.keys() if cid not in demo_results and face_data[cid].get("detected", False)]
    print(f"Clips remaining: {len(clips_to_process)}")

    dummy_labels = {
        "segmentation": torch.zeros([1, 224, 224]).to(device),
        "lnm_seg": torch.zeros([1, 5, 2]).to(device),
        "landmark": torch.zeros([1, 68, 2]).to(device),
        "headpose": torch.zeros([1, 3]).to(device),
        "attribute": torch.zeros([1, 40]).to(device),
        "a_g_e": torch.zeros([1, 3]).to(device),
        "visibility": torch.zeros([1, 29]).to(device)
    }
    task_tensor = torch.tensor([4]).to(device)

    processed_count = 0
    for clip_id in tqdm(clips_to_process, desc=f"Annotating {args.corpus}"):
        vid_info = face_data[clip_id]
        vid_path = get_video_path(args.corpus, clip_id, args.share_dir)
        
        if not vid_path.exists():
            continue

        best_frame_idx = vid_info["best_frame_idx"]
        
        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            continue

        # SCRFD Detection + VoxCeleb Padding
        pad = 100 if args.corpus == "voxceleb2_test" else 0
        if pad > 0:
            padded_frame = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[0,0,0])
            faces = app.get(padded_frame)
        else:
            faces = app.get(frame)

        if not faces:
            continue
            
        # Get largest face and adjust bbox for padding
        largest_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        x1, y1, x2, y2 = [int(v) for v in largest_face.bbox]
        x1 -= pad; y1 -= pad; x2 -= pad; y2 -= pad

        # Ensure bounds make sense after un-padding
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = image.crop((x1, y1, x2, y2))
        
        img_tensor = transforms_image(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, _, _, _, age_out, gender_out, race_out, _ = model(img_tensor, dummy_labels, task_tensor)
            age_pred = torch.argmax(age_out, dim=1)[0].item()
            gender_pred = torch.argmax(gender_out, dim=1)[0].item()
            race_pred = torch.argmax(race_out, dim=1)[0].item()

        demo_results[clip_id] = {
            "age_group": AGE_MAP.get(age_pred, str(age_pred)),
            "gender": GENDER_MAP.get(gender_pred, str(gender_pred)),
            "ethnicity": RACE_MAP.get(race_pred, str(race_pred)),
            "demo_source": "facexformer"
        }

        processed_count += 1
        
        if processed_count % args.batch_save == 0:
            with open(demo_path, "w") as f:
                json.dump(demo_results, f, indent=2)

    with open(demo_path, "w") as f:
        json.dump(demo_results, f, indent=2)
    print(f"Finished {args.corpus}. Total saved: {len(demo_results)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True, choices=["voxceleb2_test", "hdtf", "mead"])
    parser.add_argument("--workdir", type=str, default=os.environ.get("VEIL_ROOT", "veilbench_data") + "/")
    parser.add_argument("--share_dir", type=str, default=os.environ.get("SHARE_ROOT", "raw_corpora"))
    parser.add_argument("--model_path", type=str, default="./facexformer/ckpts/model.pt")
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--batch_save", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
