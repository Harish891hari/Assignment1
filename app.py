import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_transfer_model

# 1. Page Configuration
st.set_page_config(
    page_title="EcoSort AI | Recyclable Material Classifier",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS Styling for a Modern Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #2b9348;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6rem;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #55a630;
        color: white;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/?size=100&id=11352&format=png&color=2b9348", width=80)
    st.title("EcoSort AI")
    st.markdown("---")
    st.markdown("### About the Project")
    st.write("This deep learning application classifies waste materials into 6 standard recycling categories using a **MobileNetV2 Transfer Learning CNN** model.")
    
    st.markdown("### Supported Classes:")
    st.markdown("📦 **Cardboard**\n🍾 **Glass**\n🥫 **Metal**\n📄 **Paper**\n🥤 **Plastic**\n🗑️ **Trash**")
    st.markdown("---")
    st.caption("Built with PyTorch & Streamlit")

# 4. Main App Header
st.title("♻️ Recyclable Material Classification")
st.markdown("Upload a photo of waste material to instantly classify it and view its sorting guidance.")
st.markdown("---")

# 5. Model Caching & Loading
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    model = get_transfer_model(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load("transfer_recycling.pth", map_location=device))
    model.eval()
    return model, device, class_names

with st.spinner("Initializing AI model..."):
    model, device, class_names = load_model()

# 6. Digital Screenshot & Synthetic Image Detector
def detect_digital_screenshot(img: Image.Image) -> tuple[bool, str]:
    """Detects whether an image is a digital computer screenshot or synthetic UI rather than a real camera photo."""
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb).reshape(-1, 3)
    total_pixels = len(arr)
    
    # 1. Check for pure digital white (255, 255, 255) common in web UI / documents
    pure_white_count = np.sum(np.all(arr == [255, 255, 255], axis=1))
    pure_white_pct = (pure_white_count / total_pixels) * 100
    
    # 2. Check for solid flat digital color dominance
    thumb = np.array(img_rgb.resize((100, 100))).reshape(-1, 3)
    _, counts = np.unique(thumb, axis=0, return_counts=True)
    top_color_pct = (np.max(counts) / len(thumb)) * 100

    if pure_white_pct >= 8.0:
        return True, f"Image contains {pure_white_pct:.1f}% mathematically pure digital white pixels (#ffffff)."
    if top_color_pct >= 16.0:
        return True, f"Image contains {top_color_pct:.1f}% solid flat digital color fill."
        
    return False, ""

# 7. Image Preprocessing Pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 7. File Uploader Section
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.subheader("Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Target Material", width='stretch')
        
    with col2:
        st.subheader("Analysis")
        st.write("")
        st.write("")
        classify_btn = st.button("🚀 Classify Material")

    if classify_btn:
        # Pre-check: Detect whether the image is a computer screenshot or web page
        is_screenshot, reason = detect_digital_screenshot(image)
        
        if is_screenshot:
            st.markdown("---")
            st.error("### 🖥️ Digital Screenshot / Web Page Detected!")
            st.warning(
                f"**Status: 🚫 NOT PHYSICAL WASTE**\n\n"
                f"This image appears to be a **digital computer screen, website page, or UI screenshot**, not physical recyclable waste.\n\n"
                f"🔍 **Diagnostic:** {reason}\n\n"
                f"🧠 **Why did the AI previously call this 'Paper'?**\n"
                f"A web page consists of pure white backgrounds, black text lines, and rectangular form inputs. "
                f"To a neural network trained on TrashNet, these visual shapes strongly resemble a printed paper document. "
                f"This guardrail now intercepts digital graphics to prevent false classifications.\n\n"
                f"💡 **What to do:** Please upload a camera photo of actual physical waste (cans, bottles, cardboard boxes, plastics, etc.)."
            )
        else:
            with st.spinner("Analyzing material patterns..."):
                input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted.item()].upper()
            confidence_score = confidence.item() * 100
            
            st.markdown("---")
            
            # 8. Recycling Guidance Mapping
            recycling_info = {
                'cardboard': {
                    'name': '📦 Cardboard',
                    'status': '✅ RECYCLABLE', 
                    'color': 'success', 
                    'guide': 'Flatten boxes completely and keep dry before putting in the paper/cardboard bin.'
                },
                'glass': {
                    'name': '🍾 Glass',
                    'status': '✅ RECYCLABLE', 
                    'color': 'success', 
                    'guide': 'Rinse clean and remove caps/lids. Glass can be recycled infinitely.'
                },
                'metal': {
                    'name': '🥫 Metal',
                    'status': '✅ RECYCLABLE', 
                    'color': 'success', 
                    'guide': 'Rinse food or liquid residue from aluminum and steel cans before recycling.'
                },
                'paper': {
                    'name': '📄 Paper',
                    'status': '✅ RECYCLABLE', 
                    'color': 'success', 
                    'guide': 'Keep paper clean and dry. Greasy paper (like used pizza boxes) belongs in regular trash.'
                },
                'plastic': {
                    'name': '🥤 Plastic',
                    'status': '✅ RECYCLABLE (Check Local Rules)', 
                    'color': 'info', 
                    'guide': 'Rinse containers and check the resin code. Rigid bottles and jugs are widely accepted.'
                },
                'trash': {
                    'name': '🗑️ Trash',
                    'status': '❌ NON-RECYCLABLE (General Waste)', 
                    'color': 'error', 
                    'guide': 'This item cannot be recycled and should be disposed of in regular garbage/landfill.'
                }
            }

            recyclable_classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
            total_recyclable_prob = sum(probabilities[class_names.index(c)].item() for c in recyclable_classes) * 100

            # Rank classes by score
            sorted_indices = torch.argsort(probabilities, descending=True)
            top1_idx = sorted_indices[0].item()
            top2_idx = sorted_indices[1].item()
            top3_idx = sorted_indices[2].item()

            top1_key = class_names[top1_idx]
            top1_score = probabilities[top1_idx].item() * 100
            top2_key = class_names[top2_idx]
            top2_score = probabilities[top2_idx].item() * 100
            top3_key = class_names[top3_idx]
            top3_score = probabilities[top3_idx].item() * 100

            # --- CASE 1: Single Dominant Material (>= 60%) ---
            if top1_score >= 60.0:
                info = recycling_info[top1_key]
                if top1_score >= 80.0:
                    st.success(f"### Identified Material: **{info['name']}**")
                else:
                    st.info(f"### Predominant Material: **{info['name']}** (Moderate Confidence)")
                    
                st.metric(label="Confidence", value=f"{top1_score:.2f}%")
                
                if info['color'] == 'success':
                    st.success(f"**Status: {info['status']}**\n\n💡 **Tip:** {info['guide']}")
                elif info['color'] == 'info':
                    st.info(f"**Status: {info['status']}**\n\n💡 **Tip:** {info['guide']}")
                else:
                    st.error(f"**Status: {info['status']}**\n\n💡 **Tip:** {info['guide']}")

            # --- CASE 2: Mixed Recyclable Waste (Pile / Multi-Material) ---
            # When multiple recyclable materials co-exist (e.g. Metal + Paper or Plastic in a heap)
            elif (
                total_recyclable_prob >= 75.0 
                and top1_key in recyclable_classes 
                and top2_key in recyclable_classes 
                and top2_score >= 18.0
            ):
                st.success("### ♻️ Mixed Recyclable Materials Detected!")
                
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(label="Total Recyclability Score", value=f"{total_recyclable_prob:.1f}%")
                with col_m2:
                    st.metric(label="Primary Material", value=f"{top1_key.capitalize()} ({top1_score:.1f}%)")
                    
                st.info(
                    f"**Status: 🔄 MIXED RECYCLABLES — SEGREGATION REQUIRED**\n\n"
                    f"This image contains a **mixture of different recyclable items** rather than a single item:\n\n"
                    f"1. {recycling_info[top1_key]['name']}: **{top1_score:.1f}%**\n"
                    f"2. {recycling_info[top2_key]['name']}: **{top2_score:.1f}%**\n"
                    + (f"3. {recycling_info[top3_key]['name']}: **{top3_score:.1f}%**\n" if top3_score >= 8.0 and top3_key in recyclable_classes else "")
                    + f"\n💡 **Sorting Guidance:** Before recycling, **separate materials into their respective bins**:\n"
                    f"- *{top1_key.capitalize()}:* {recycling_info[top1_key]['guide']}\n"
                    f"- *{top2_key.capitalize()}:* {recycling_info[top2_key]['guide']}"
                )

            # --- CASE 3: Single Moderate Signal (>= 50% recyclable) ---
            elif top1_score >= 50.0 and top1_key in recyclable_classes and total_recyclable_prob >= 80.0:
                info = recycling_info[top1_key]
                st.info(f"### Likely Material: **{info['name']}**")
                st.metric(label="Confidence", value=f"{top1_score:.2f}%")
                st.info(f"**Status: {info['status']}**\n\n💡 **Tip:** {info['guide']}")

            # --- CASE 4: Unrecognized / Non-Waste Object / Screenshot ---
            else:
                st.error("### ⚠️ No Clear Recyclable Material Detected")
                st.metric(label="Top Pattern Match", value=f"{top1_key.capitalize()} ({top1_score:.2f}%)")
                
                st.warning(
                    f"**Status: ❓ UNCERTAIN / UNRECOGNIZED OBJECT**\n\n"
                    f"The model could not detect a distinct recyclable item or identifiable mixed waste pile.\n\n"
                    f"📌 **Note:** The AI is trained on **physical recyclable items**. Digital screenshots, document pages, or non-waste items will not produce a valid classification.\n\n"
                    f"💡 **Tip:** Please upload a clear photo of physical waste items."
                )

            # Confidence Breakdown Progress Bars
            st.write("#### 📊 Confidence Breakdown Across All Classes:")
            for idx, name in enumerate(class_names):
                prob_val = probabilities[idx].item()
                st.progress(int(prob_val * 100), text=f"{name.capitalize()}: {prob_val * 100:.2f}%")
else:
    st.info("👆 Upload an image using the file uploader above to get started.")