# Zero-Shot-SAM3Dfy

## 1. Dataset Installation

### AeroPath Dataset

Download the dataset from:
```
https://zenodo.org/records/10069289
```

### BCTV and MSD

```
python download_data.py
```

#### Expected File Structure
```
data/
└── AeroPath/
    ├── images/
    │   ├── case_001.nii.gz
    │   ├── case_002.nii.gz
    │   └── ...
    └── labels/
        ├── case_001.nii.gz
        ├── case_002.nii.gz
        └── ...
```

## 2. Hunyuan3D Installation

### Clone and Install Dependencies

```bash
cd models
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git
cd Hunyuan3D-2
pip install -r requirements.txt
pip install -e .
```

### Download Model Weights

Download the weights from Hugging Face:
```
https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-dit-v2-1
```

Place the downloaded weights in the appropriate directory within `models/Hunyuan3D-2/`.

For more information, see the official repository: https://github.com/Tencent-Hunyuan/Hunyuan3D-2


## Running

### For Hunyuan:

Single Run
```
python run.py --config config/AeroPath_config.yaml
```

Multi-run
```
python run.py --config config/AeroPath_config.yaml config/BCTV_config.yaml config/MSD_config.yaml
```

