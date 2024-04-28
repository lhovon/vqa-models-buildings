# VQA models and classifiers to find buildings in images

Download and unzip the following files:
- Dataset: https://buildings-img-classifier-trained-models.s3.amazonaws.com/data.zip
- Pretrained classifer models, so you can skip training: https://buildings-img-classifier-trained-models.s3.amazonaws.com/tmp.zip 

You should end-up with a `data` and `tmp` directories in this folder.

Install dependencies:
```
# Optionally create a virtual environment
pip -m venv .venv
.venv\Scripts\activate # windows
source .venv/bin/activate # mac/linux

pip install -r requirements.txt
```

`vqa_measure_sensi_speci.py` will measure the sensitivity and specificity of 2 VQA models (BLIP and ViLT) when answering questions to identify buildings and metal buildings in pictrues.

`brails_train_and_eval_classifiers.py` will train and evaluate image classifiers from [BRAILS](https://nheri-simcenter.github.io/BRAILS-Documentation/). You can download my trained models here to skip training, which could take a long time (days) if you only have CPUs: .

Then unzip it, you should end up with a `tmp` folder.

`vqa_eval_models_as_classifiers.py` will use the empirically measured sensi/speci and a set probability threshold (.5) to use the VQA models as classifiers, so we can compare them to the BRAILS models.