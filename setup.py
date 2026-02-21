from setuptools import setup, find_packages

setup(
    name="hcn-ta",
    version="1.0.0",
    author="Taiba Majid Wani, Madleen Uceker, Farooq Ahmad Wani, Irene Amerini",
    author_email="majid@diag.uniroma1.it",
    description="HCN-TA: Hierarchical Capsule Network with Temporal Attention for Audio Deepfake Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/HCN-TA",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0", "torchaudio>=0.12.0", "torchvision>=0.13.0",
        "numpy>=1.21.0", "librosa>=0.9.0", "scikit-learn>=1.0.0",
        "tqdm>=4.62.0", "pyyaml>=6.0", "noisereduce>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
