# Document Image Binarization & Text Enhancement
A production-ready Python library for document image binarization optimized for OCR. Convert shadow-removed grayscale/color images into high-contrast binary images with preserved text quality.

🎯 Features
- 10+ Binarization Algorithms: Global, adaptive, and advanced methods
- Unified Pipeline: Easy-to-use interface with configurable stages
- Post-Processing: Morphological operations, component filtering, border removal
- Text Enhancement: Stroke normalization, character connection/separation
- OCR Integration: Built-in Tesseract integration with accuracy metrics
- Parameter Optimization: Automatic tuning using Bayesian optimization
- Evaluation Framework: Comprehensive metrics (CER, WER, PSNR, SSIM, IoU)

📦 Installation
- From Source:
git clone https://github.com/giabao-github/document-binarization.git
cd document-binarization
pip install -e .

- Dependencies:
pip install -r requirements.txt