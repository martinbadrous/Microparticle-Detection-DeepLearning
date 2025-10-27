# 🔬 Microparticle Counting & Tracking

A **modern OpenCV + Python pipeline** for detecting, counting, and tracking microscopic particles (e.g., microplastics, biological cells, or industrial contaminants) in image or video streams.

Developed and maintained by **[Martin Badrous](https://github.com/martinbadrous)**.

---

## 🌍 Overview

This repository provides a complete, modular, and CLI-based framework to process microscopy data for **particle counting and motion tracking**.

It supports:
- ✅ Real-time analysis from video or live camera feeds  
- ✅ Batch processing of folders of microscope images  
- ✅ Reproducible per-frame counts and aggregated metrics  
- ✅ Lightweight dependency footprint (OpenCV + NumPy + pandas)  

Applications include:
- 🧫 **Environmental research** – microplastics or microbe detection  
- ⚗️ **Biological imaging** – particle motion analysis  
- 🧰 **Industrial inspection** – contamination or filter validation  

---

## ⚙️ Features

| Category | Description |
|-----------|--------------|
| **Flexible Input** | Accepts single videos, webcam streams, or folders of images |
| **Detection Options** | Hough-circle or contour-based particle detection |
| **Online Tracking** | Centroid-based tracker with aging to prevent double counting |
| **Logging & Output** | Saves per-frame counts to CSV and summary metrics to JSON |
| **Visualization** | Optional annotated MP4 export or per-frame image saving |
| **Command-Line Interface** | Fully configurable parameters for reproducibility |

---

## 🗂 Repository Structure

```bash
Microparticle-counting/
├── microparticle_pipeline.py       # Main pipeline script (modernized)
├── requirements_microparticles.txt # Dependencies
├── DEMO_PARTICLES/                 # Example image/video data
└── outputs/
    └── exp1/
        ├── counts.csv              # Per-frame particle counts
        ├── summary.json            # Summary metrics
        ├── annotated.mp4           # Optional annotated video
        └── frame_000123.jpg        # Saved annotated frames (optional)
```

---

## 🚀 Quick Setup & Usage

💡 Copy and paste the following commands into your terminal.

```bash
# 1. Clone the repository
git clone https://github.com/martinbadrous/Microparticle-counting.git
cd Microparticle-counting

# 2. Install dependencies
pip install -r requirements_microparticles.txt

# 3. Run on a video using Hough detector
python microparticle_pipeline.py   --input path/to/video.mp4   --detector hough   --output_dir outputs/exp1   --save_video   --hough_dp 1.2 --hough_min_dist 12   --hough_param1 120 --hough_param2 15   --hough_min_radius 3 --hough_max_radius 20

# 4. Run on an image folder with contour detection
python microparticle_pipeline.py   --input path/to/image_folder   --detector contour   --output_dir outputs/exp2   --contour_thresh 25 --morph_kernel 3

# 5. Live webcam stream (ESC to exit)
python microparticle_pipeline.py --input 0 --detector hough --display
```

---

## 🧩 Detection Methods

### 🔵 Hough Circle Detector
Detects circular particles efficiently in well-contrasted microscope images.

| Parameter | Description | Default |
|------------|--------------|----------|
| `--hough_dp` | Inverse ratio of accumulator resolution | 1.2 |
| `--hough_min_dist` | Minimum distance between detected circles | 12 |
| `--hough_param1` | Canny high threshold | 120 |
| `--hough_param2` | Accumulator threshold for center detection | 15 |
| `--hough_min_radius` | Minimum circle radius | 3 |
| `--hough_max_radius` | Maximum circle radius | 20 |

### ⚫ Contour Detector
Detects irregular particle shapes using thresholding and morphology.

| Parameter | Description | Default |
|------------|--------------|----------|
| `--contour_thresh` | Binary threshold value | 25 |
| `--morph_kernel` | Morphology kernel size | 3 |
| `--min_area` | Minimum contour area | 8.0 |
| `--max_area` | Maximum contour area | 1e6 |
| `--adaptive_thresh` | Use adaptive thresholding | (flag) |

---

## 📊 Output Files

| File | Description |
|------|--------------|
| `counts.csv` | Per-frame active particle count |
| `summary.json` | Aggregate summary of counts and parameters |
| `annotated.mp4` | Annotated video (if `--save_video` used) |
| `frame_XXXXX.jpg` | Annotated frame outputs (if `--save_frames` used) |

Example of `summary.json`:
```json
{
  "frames_processed": 512,
  "total_unique_ids": 298,
  "avg_active_per_frame": 23.4,
  "detector": "hough"
}
```

---

## 📈 Example Output

**Terminal Log**
```
Frame 125: 21 particles detected
Frame 126: 23 particles detected
...
Saved per-frame counts to: outputs/exp1/counts.csv
Saved summary to: outputs/exp1/summary.json
Annotated video: outputs/exp1/annotated.mp4
```

**Visualization Example**

Each particle is annotated with its ID and boundary, with live count displayed in the top left.

---

## 🧰 Requirements

```bash
opencv-python>=4.7.0
numpy>=1.23.0
pandas>=1.5.0
```

---

## 🧠 Tips for Best Results

- Ensure **good contrast** and **consistent illumination** in microscope captures.  
- Tune detection parameters for particle size range.  
- Apply **Gaussian blur** for noise reduction.  
- Use **adaptive thresholding** for uneven backgrounds.  
- For large particles or irregular shapes, try `--detector contour`.

---

## 🧭 Roadmap

- [ ] Add YOLOv8-based deep detector option  
- [ ] Add 3D particle tracking (z-stack / time-series support)  
- [ ] Add Streamlit dashboard for visualization  
- [ ] Integrate CSV/Excel export for research reporting  

---

## 👨‍💻 Author

**Martin Badrous**  
Computer Vision & Robotics Engineer  
🎓 M.Sc. in Computer Vision and Robotics (Université de Bourgogne)  
📍 Based in France | 🇪🇬 Egyptian origin  
📧 martin.badrous@gmail.com  
🔗 [GitHub](https://github.com/martinbadrous) | [LinkedIn](https://linkedin.com/in/martinbadrous)

---

⭐ **If this project helps your research or work, please give it a star on GitHub!**
