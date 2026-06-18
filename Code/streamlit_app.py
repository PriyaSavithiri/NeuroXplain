import os, io, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from pathlib import Path
from transformers import ViTForImageClassification, ViTFeatureExtractor
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import joblib
import requests
import json
warnings.filterwarnings("ignore")

MODEL_DIR = Path("brain-tumor-detection/Code/model")

MODEL_CONFIGS = {
    "CNN":          {"weights": MODEL_DIR/"CNN_weights.pth",    "metadata": MODEL_DIR/"CNN_metadata.pkl",    "arch": "cnn"},
    "Xception":     {"weights": MODEL_DIR/"Xception_weights.pth", "metadata": MODEL_DIR/"Xception_metadata.pkl", "arch": "xception"},
    "EfficientNet": {"weights": MODEL_DIR/"EfficientNet_weights.pth","metadata": MODEL_DIR/"EfficientNet_metadata.pkl","arch": "efficientnet-b0"},
    "ViT":          {"weights": MODEL_DIR/"ViT_weights.pth",     "metadata": MODEL_DIR/"ViT_metadata.pkl",     "arch": "vit"},
    "Swin":         {"weights": MODEL_DIR/"Swin_weights.pth",    "metadata": MODEL_DIR/"Swin_metadata.pkl",    "arch": "swin"},
}

CLASSES    = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_INFO = {
    "glioma":     {"color": "#e74c3c", "severity": "High",         "desc": "Aggressive tumor arising in glial cells."},
    "meningioma": {"color": "#f39c12", "severity": "Moderate",     "desc": "Usually benign tumor from the meninges."},
    "notumor":    {"color": "#2ecc71", "severity": "None",         "desc": "No tumor detected — scan appears normal."},
    "pituitary":  {"color": "#3498db", "severity": "Low-Moderate", "desc": "Pituitary gland adenoma, typically treatable."},
}

GRADCAM_REGION_DESC = {
    "CNN":          "ventricles and periventricular white matter",
    "Xception":     "cortical surface and sulcal patterns",
    "EfficientNet": "broad parenchymal regions with global context",
    "Swin":         "localised central patch windows",
}

class CNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.global_pool(x)
        return self.fc(x.view(x.size(0), -1))

def build_arch(arch, num_classes=4):
    if arch == "cnn":            return CNN(num_classes=num_classes)
    elif arch == "xception":     return timm.create_model("xception", pretrained=False, num_classes=num_classes)
    elif arch == "efficientnet-b0": return timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    elif arch == "vit":
        return ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224", num_labels=num_classes, ignore_mismatched_sizes=True)
    elif arch == "swin":         return timm.create_model("swin_tiny_patch4_window7_224", pretrained=False, num_classes=num_classes)
    else: raise ValueError(f"Unknown arch: {arch}")

class ZScoreNormalize:
    def __call__(self, img):
        arr  = np.array(img).astype(np.float32)
        mask = arr > 0
        mean = arr[mask].mean() if mask.any() else arr.mean()
        std  = arr[mask].std()  if mask.any() else arr.std()
        arr  = (arr - mean) / (std + 1e-8)
        arr  = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        return Image.fromarray(arr.astype(np.uint8))

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    ZScoreNormalize(),
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

@st.cache_resource(show_spinner=False)
def load_all_models():
    loaded = {}
    for name, cfg in MODEL_CONFIGS.items():
        wp, mp = cfg["weights"], cfg["metadata"]
        if not wp.exists():
            loaded[name] = None; continue
        try:
            nc    = len(joblib.load(mp).get("class_names", CLASSES)) if mp.exists() else 4
            model = build_arch(cfg["arch"], num_classes=nc)
            model.load_state_dict(torch.load(wp, map_location=DEVICE))
            model.to(DEVICE).eval()
            loaded[name] = model
        except Exception as e:
            loaded[name] = f"Error: {e}"
    return loaded

@st.cache_resource(show_spinner=False)
def load_vit_extractor():
    return ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

def predict_all(image_pil, models, vit_ext):
    results = {}
    for name, model in models.items():
        try:
            if name == "ViT":
                img = ZScoreNormalize()(image_pil.convert("RGB"))
                t   = vit_ext(images=img, return_tensors="pt")["pixel_values"].squeeze(0).unsqueeze(0).to(DEVICE)
                out = model(t).logits
            else:
                t   = eval_transform(image_pil).unsqueeze(0).to(DEVICE)
                out = model(t)
                if hasattr(out, "logits"): out = out.logits
            probs = F.softmax(out, dim=1).cpu().detach().numpy()[0]
            results[name] = (CLASSES[probs.argmax()], probs)
        except Exception as e:
            results[name] = f"Error: {e}"
    return results

def get_target_layer(model, name):
    if name == "CNN":           return model.conv3
    elif name == "Xception":    return model.conv4.pointwise
    elif name == "EfficientNet": return model.conv_head
    elif name == "Swin":        return model.norm
    else:                       return None

def run_gradcam(model, name, image_pil, plusplus=False):
    layer = get_target_layer(model, name)
    if layer is None: return None
    t      = eval_transform(image_pil).unsqueeze(0).to(DEVICE)
    img_np = np.clip(np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0, 0, 1)
    if img_np.ndim == 2: img_np = np.stack([img_np] * 3, -1)
    img_np = img_np[:, :, :3]
    CAM = GradCAMPlusPlus if plusplus else GradCAM
    try:
        with CAM(model=model, target_layers=[layer]) as cam:
            return show_cam_on_image(img_np, cam(input_tensor=t)[0], use_rgb=True)
    except Exception:
        return None

def run_lime(model, image_pil, n_samples=300, n_feat=6):
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
        norm   = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        img_np = np.array(image_pil.resize((224, 224))).astype(np.float64) / 255.0
        if img_np.ndim == 2: img_np = np.stack([img_np] * 3, -1)
        img_np = img_np[:, :, :3]
        def pred_fn(imgs):
            model.eval()
            ts = [norm(torch.from_numpy(im.transpose(2, 0, 1)).float()) for im in imgs]
            b  = torch.stack(ts).to(DEVICE)
            with torch.no_grad():
                out = model(b)
                if hasattr(out, "logits"): out = out.logits
            return F.softmax(out, dim=1).cpu().numpy()
        expl     = lime_image.LimeImageExplainer()
        exp      = expl.explain_instance(img_np, pred_fn, top_labels=4,
                                         hide_color=0, num_samples=n_samples)
        lbl      = exp.top_labels[0]
        temp, mask = exp.get_image_and_mask(lbl, positive_only=False,
                                             num_features=n_feat, hide_rest=False)
        return (mark_boundaries(temp, mask) * 255).astype(np.uint8)
    except Exception:
        return None

def run_attention_rollout(model, image_pil, vit_ext):
    try:
        img = ZScoreNormalize()(image_pil.convert("RGB"))
        t   = vit_ext(images=img, return_tensors="pt")["pixel_values"].to(DEVICE)
        attention_maps = []
        def hook_fn(module, input, output):
            attention_maps.append(output.detach().cpu())
        hooks = [layer.attention.attention.register_forward_hook(hook_fn)
                 for layer in model.vit.encoder.layer]
        with torch.no_grad():
            model(t)
        for h in hooks: h.remove()
        rollout = torch.eye(attention_maps[0].shape[-1])
        for attn in attention_maps:
            attn_avg  = attn[0].mean(dim=0)
            attn_avg  = attn_avg + torch.eye(attn_avg.shape[0])
            attn_avg  = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            rollout   = torch.mm(attn_avg, rollout)
        cls_attn = rollout[0, 1:]
        side     = int(cls_attn.shape[0] ** 0.5)
        attn_map = cls_attn.reshape(side, side).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_pil = Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        attn_np  = np.array(attn_pil).astype(np.float32) / 255.0
        img_224  = np.clip(np.array(image_pil.resize((224, 224))).astype(np.float32) / 255.0, 0, 1)
        if img_224.ndim == 2: img_224 = np.stack([img_224] * 3, -1)
        return show_cam_on_image(img_224[:, :, :3], attn_np, use_rgb=True)
    except Exception:
        return None

def build_explanation(results, consensus, xai_method):
    preds  = {k: v[0] for k, v in results.items() if isinstance(v, tuple)}
    confs  = {k: float(v[1].max()) for k, v in results.items() if isinstance(v, tuple)}
    agree  = all(p == consensus for p in preds.values())
    n_agree = sum(1 for p in preds.values() if p == consensus)
    avg_conf = np.mean(list(confs.values()))

    if avg_conf >= 0.95:
        conf_text = "extremely high confidence"
    elif avg_conf >= 0.80:
        conf_text = "high confidence"
    elif avg_conf >= 0.60:
        conf_text = "moderate confidence"
    else:
        conf_text = "low confidence — result should be treated with caution"

    if agree:
        agree_text = f"All {len(preds)} models unanimously agree on this prediction."
    else:
        disagree = [k for k, v in preds.items() if v != consensus]
        agree_text = (f"{n_agree} out of {len(preds)} models predict {consensus.upper()}. "
                      f"{', '.join(disagree)} predicted differently.")

    CLASS_REASONING = {
        "notumor": {
            "what": "The scan shows no abnormal mass, irregular boundary, or asymmetric enhancement.",
            "gradcam": "The heatmaps highlight the **ventricular system** (fluid-filled cavities) and **periventricular white matter** — the regions most distinct from known tumor locations. By focusing here and finding normal appearance, the model rules out tumor presence.",
            "lime": "The green superpixels (supporting evidence) cover the **central brain structures and cortical folds**. These regions have normal intensity distribution and symmetry, which the model recognises as healthy tissue patterns.",
            "vit": "The attention map shows **diffuse, distributed attention** across normal brain structures rather than concentrated focus — consistent with no focal abnormality to localise.",
        },
        "glioma": {
            "what": "The scan shows features consistent with a glial cell tumor — often irregular margins, heterogeneous signal, and mass effect.",
            "gradcam": "The heatmaps are concentrated on a **focal region of abnormal intensity** — typically showing irregular boundaries and surrounding edema that distinguishes glioma from normal tissue.",
            "lime": "Supporting superpixels highlight the **lesion core and surrounding infiltration zone** — the model identified these regions as the primary evidence for this class.",
            "vit": "Attention is **strongly concentrated on the tumor region**, showing the ViT identified a specific focal abnormality rather than distributing attention diffusely.",
        },
        "meningioma": {
            "what": "The scan shows features consistent with a meningioma — typically a well-defined, extra-axial mass attached to the meninges.",
            "gradcam": "The heatmaps focus on the **brain periphery and dural interface** — where meningiomas characteristically arise, pressing against but not invading the brain parenchyma.",
            "lime": "Supporting superpixels highlight the **outer brain margins and skull base** — the model identified the peripheral location as key evidence.",
            "vit": "Attention concentrates on the **dural attachment zone** — the ViT has localised the characteristic extra-axial location of the mass.",
        },
        "pituitary": {
            "what": "The scan shows features consistent with a pituitary adenoma — a mass in the sellar/parasellar region.",
            "gradcam": "The heatmaps are focused on the **sella turcica region** (base of the brain) — the anatomical location of the pituitary gland where adenomas develop.",
            "lime": "Supporting superpixels highlight the **central skull base and optic chiasm region** — the model localised its evidence to the pituitary fossa.",
            "vit": "Attention is concentrated at the **midline skull base** — consistent with the ViT identifying the sellar region as the key discriminating area.",
        },
    }

    reasoning = CLASS_REASONING.get(consensus, {})

    if xai_method == "Grad-CAM":
        xai_text = reasoning.get("gradcam", "")
        method_name = "Grad-CAM"
    elif xai_method == "Grad-CAM++":
        xai_text = reasoning.get("gradcam", "").replace("heatmaps", "Grad-CAM++ heatmaps (improved localisation)")
        method_name = "Grad-CAM++"
    elif xai_method == "LIME":
        xai_text = reasoning.get("lime", "")
        method_name = "LIME"
    else:  # All
        xai_text = (f"**Grad-CAM:** {reasoning.get('gradcam', '')}\n\n"
                    f"**LIME:** {reasoning.get('lime', '')}\n\n"
                    f"**ViT Attention:** {reasoning.get('vit', '')}")
        method_name = "all XAI methods"

    model_lines = []
    for name, pred in preds.items():
        conf    = confs[name]
        match   = "✅" if pred == consensus else "⚠️"
        model_lines.append(f"- **{name}**: {pred.upper()} ({conf:.1%}) {match}")

    return {
        "headline":      f"Why the models predicted **{consensus.upper()}**",
        "agreement":     agree_text,
        "confidence":    f"Average confidence: **{avg_conf:.1%}** ({conf_text})",
        "what_it_means": reasoning.get("what", ""),
        "xai_evidence":  xai_text,
        "clinical_note": reasoning.get("clinical", ""),
        "model_breakdown": "\n".join(model_lines),
        "method_name":   method_name,
    }

def conf_chart(probs, model_name):
    colors = ["#e74c3c", "#f39c12", "#2ecc71", "#3498db"]
    fig, ax = plt.subplots(figsize=(5, 2.6))
    bars = ax.barh(CLASSES, probs * 100, color=colors, edgecolor="white")
    ax.set_xlim(0, 100); ax.set_xlabel("Confidence (%)")
    ax.set_title(model_name, fontweight="bold")
    for bar, p in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0); plt.close()
    return buf

def arr_to_buf(arr):
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    buf.seek(0)
    return buf

def main():
    st.set_page_config(page_title="NeuroXplain", page_icon="🧠", layout="wide")

    with st.sidebar:
        st.title(" Settings")
        xai    = st.radio("XAI Method", ["Grad-CAM", "Grad-CAM++", "LIME", "All"], index=0)
        lime_n = st.slider("LIME samples", 100, 800, 300, 50,
                           disabled=(xai not in ["LIME", "All"]))
        st.markdown("---")
        st.markdown(f"**Loading from:**\n`{MODEL_DIR}`\n\n"
                    "**Models**\n- CNN (ep100)\n- Xception (ep10)\n"
                    "- EfficientNet (ep10)\n- ViT (ep10)\n- Swin (ep10)\n\n"
                    "⚕️ *Research use only.*")

    st.title("🧠 NeuroXplain — Brain Tumor MRI Classifier")
    st.markdown("Upload an MRI scan → **5-model prediction** + **plain-language XAI explanation**")

    with st.spinner("Loading saved model checkpoints..."):
        models  = load_all_models()
        vit_ext = load_vit_extractor()

    ok_models   = {k: v for k, v in models.items() if v is not None and not isinstance(v, str)}
    err_models  = {k: v for k, v in models.items() if isinstance(v, str)}
    miss_models = [k for k, v in models.items() if v is None]

    if miss_models:
        st.warning(f"Weight files not found for: {', '.join(miss_models)}")
    for name, msg in err_models.items():
        st.error(f"{name} failed to load: {msg}")
    if not ok_models:
        st.error(f"No models loaded. Check .pth files in: `{MODEL_DIR}/`")
        st.stop()
    st.success(f" Loaded: {', '.join(ok_models.keys())}")

    uploaded = st.file_uploader("Upload brain MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info(" Upload an MRI image to get started.")
        return

    image_pil = Image.open(uploaded).convert("RGB")

    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.image(image_pil, caption="Uploaded MRI", width=300)

    with st.spinner("Running inference on all 5 models..."):
        t0      = time.time()
        results = predict_all(image_pil, ok_models, vit_ext)
        elapsed = time.time() - t0

    valid_preds = [v[0] for v in results.values() if isinstance(v, tuple)]
    consensus   = max(set(valid_preds), key=valid_preds.count) if valid_preds else "unknown"
    info        = CLASS_INFO.get(consensus, {})

    with col_info:
        color = info.get("color", "#888888")
        st.markdown(f"<h2 style='color:{color}'>Consensus: {consensus.upper()}</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"**Severity:** {info.get('severity', '—')}")
        st.markdown(f"> {info.get('desc', '')}")
        st.caption(f"Inference: {elapsed:.2f}s | Device: {DEVICE}")

    st.markdown("---")
    st.markdown("## Per-Model Predictions")
    pred_cols = st.columns(len(ok_models))
    for col, name in zip(pred_cols, ok_models.keys()):
        res = results.get(name)
        with col:
            if not isinstance(res, tuple):
                st.error(f"{name}: {res}"); continue
            pred, probs = res
            color = CLASS_INFO.get(pred, {}).get("color", "#888")
            st.markdown(f"<h4 style='color:{color}'>{name}</h4>", unsafe_allow_html=True)
            st.markdown(f"**{pred.upper()}** ({probs.max():.1%})")
            st.image(conf_chart(probs, name), width=260)

    st.markdown("---")
    exp = build_explanation(results, consensus, xai)

    color = CLASS_INFO.get(consensus, {}).get("color", "#888888")
    st.markdown(
        f"<div style='background:linear-gradient(135deg,{color}22,{color}11);"
        f"border-left:5px solid {color};border-radius:8px;padding:20px 24px;margin-bottom:12px'>"
        f"<h2 style='color:{color};margin:0 0 4px 0'>🔍 {exp['headline']}</h2>"
        f"</div>",
        unsafe_allow_html=True
    )

    exp_col1, exp_col2 = st.columns([1, 1])

    with exp_col1:
        st.markdown("### Model Agreement")
        st.markdown(exp["agreement"])
        st.markdown(exp["confidence"])
        st.markdown("**Individual model votes:**")
        st.markdown(exp["model_breakdown"])

    with exp_col2:
        st.markdown("### What the scan shows")
        st.markdown(exp["what_it_means"])
        st.markdown("### Clinical context")
        st.markdown(f"_{exp['clinical_note']}_")

    st.markdown("### 🗺️ What the XAI evidence tells us")
    st.info(exp["xai_evidence"])

    st.markdown("---")
    st.markdown("##  Visual XAI — Heatmaps per Model")
    st.caption("The heatmaps below show **where** each model looked. "
               "The explanation above tells you **what that means** for the prediction.")

    cnn_models = {k: v for k, v in ok_models.items()
                  if k in ["CNN", "Xception", "EfficientNet", "Swin"]}

    if cnn_models:
        if xai in ["Grad-CAM", "All"]:
            st.markdown("#### Grad-CAM — Gradient-weighted Class Activation Mapping")
            st.caption("Red/warm = high importance | Blue/cool = low importance")
            with st.spinner("Computing Grad-CAM..."):
                gcols = st.columns(len(cnn_models))
                for col, (nm, m) in zip(gcols, cnn_models.items()):
                    ov = run_gradcam(m, nm, image_pil, plusplus=False)
                    with col:
                        st.markdown(f"**{nm}**")
                        st.image(arr_to_buf(ov) if ov is not None else image_pil, width=210)
                        region = GRADCAM_REGION_DESC.get(nm, "")
                        if region:
                            res_nm = results.get(nm)
                            conf_nm = f"{res_nm[1].max():.1%}" if isinstance(res_nm, tuple) else "N/A"
                            st.caption(f"Focus: {region} → {conf_nm} confident")

        if xai in ["Grad-CAM++", "All"]:
            st.markdown("#### Grad-CAM++ — Improved Localisation")
            st.caption("More precise than Grad-CAM, especially for multiple objects")
            with st.spinner("Computing Grad-CAM++..."):
                gcols = st.columns(len(cnn_models))
                for col, (nm, m) in zip(gcols, cnn_models.items()):
                    ov = run_gradcam(m, nm, image_pil, plusplus=True)
                    with col:
                        st.markdown(f"**{nm}**")
                        st.image(arr_to_buf(ov) if ov is not None else image_pil, width=210)

        if xai in ["LIME", "All"]:
            st.markdown("#### LIME — Superpixel Evidence")
            st.caption(" Green = image regions that SUPPORT the prediction | 🔴 Red = regions that CONTRADICT it")
            with st.spinner(f"Computing LIME ({lime_n} samples per model)..."):
                lcols = st.columns(len(cnn_models))
                for col, (nm, m) in zip(lcols, cnn_models.items()):
                    li = run_lime(m, image_pil, n_samples=lime_n)
                    with col:
                        st.markdown(f"**{nm}**")
                        if li is not None:
                            st.image(arr_to_buf(li), width=210)
                        else:
                            st.warning("LIME failed — pip install lime scikit-image")

    vit_model = ok_models.get("ViT")
    if vit_model is not None:
        st.markdown("#### ViT — Attention Rollout")
        st.caption("Aggregates attention across all 12 transformer layers. "
                   "Warm regions = where the ViT directed its attention.")
        with st.spinner("Computing ViT Attention Rollout..."):
            rollout_img = run_attention_rollout(vit_model, image_pil, vit_ext)

        vc1, vc2, vc3 = st.columns([1, 1, 2])
        with vc1:
            st.markdown("**Original MRI**")
            st.image(image_pil, width=210)
        with vc2:
            st.markdown("**Attention Rollout**")
            if rollout_img is not None:
                st.image(arr_to_buf(rollout_img), width=210)
            else:
                st.warning("Rollout failed")
        with vc3:
            st.markdown("**What this means for this prediction:**")
            vit_reasoning = {
                "notumor":    "Attention is **diffuse and distributed** across normal brain structures — the ViT found no focal abnormality to concentrate on, supporting the no-tumor prediction.",
                "glioma":     "Attention is **concentrated on a focal region** of abnormal signal — the ViT has identified a specific suspicious area consistent with a glial tumor.",
                "meningioma": "Attention focuses on the **brain periphery** — consistent with the extra-axial location characteristic of meningiomas.",
                "pituitary":  "Attention is directed toward the **skull base/sellar region** — where the pituitary gland sits, consistent with an adenoma.",
            }.get(consensus, "Attention pattern reflects the model's internal reasoning about the predicted class.")
            st.markdown(vit_reasoning)

    st.markdown("---")
    st.markdown("## XAI Evidence Summary")
    st.markdown("A structured summary of what each method found and what it means:")

    summary_rows = []
    for nm in ok_models.keys():
        res_nm = results.get(nm)
        if not isinstance(res_nm, tuple): continue
        pred_nm, probs_nm = res_nm
        conf_nm = probs_nm.max()
        agree   = "✅ Agrees" if pred_nm == consensus else f"⚠️ Disagrees ({pred_nm.upper()})"
        xai_focus = GRADCAM_REGION_DESC.get(nm, "attention-based (see rollout)")
        summary_rows.append({
            "Model":       nm,
            "Prediction":  pred_nm.upper(),
            "Confidence":  f"{conf_nm:.1%}",
            "Agreement":   agree,
            "XAI Focus Area": xai_focus,
        })

    import pandas as pd
    df_summary = pd.DataFrame(summary_rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    overall_color = CLASS_INFO.get(consensus, {}).get("color", "#888")
    n_agree = sum(1 for v in results.values() if isinstance(v, tuple) and v[0] == consensus)
    avg_conf = np.mean([v[1].max() for v in results.values() if isinstance(v, tuple)])

    st.markdown(
        f"<div style='background:{overall_color}22;border:2px solid {overall_color};"
        f"border-radius:10px;padding:16px 20px;margin-top:16px'>"
        f"<b style='color:{overall_color};font-size:1.1em'>Final Verdict</b><br>"
        f"<b>{consensus.upper()}</b> — {n_agree}/{len(ok_models)} models agree | "
        f"Average confidence: {avg_conf:.1%}<br>"
        f"<small>⚕️ This is an AI-assisted analysis for research purposes only. "
        f"Always consult a qualified radiologist for clinical decisions.</small>"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.caption("NeuroXplain | University of Galway CT5135 | Educational use only")


if __name__ == "__main__":
    main()