import csv
import time
import torch

from PIL import Image
from pathlib import Path
from sklearn import metrics
from torchvision import transforms
from brails.modules import ImageClassifier

from vqa_eval_models_as_classifiers import get_real_answer

def true_positive(y_true, y_pred):
    
    tp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt in ['metal', 'not_metal'] and yt == yp:
            tp += 1
    
    return tp

def true_negative(y_true, y_pred):
    
    tn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 0:
            tn += 1
            
    return tn

def false_positive(y_true, y_pred):
    
    fp = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 0 and yp == 1:
            fp += 1
            
    return fp

def false_negative(y_true, y_pred):
    
    fn = 0
    
    for yt, yp in zip(y_true, y_pred):
        
        if yt == 1 and yp == 0:
            fn += 1
            
    return fn


def train_models(model_architectures):

    for model_arch in model_architectures:

        print(f"Training {model_arch}")
        # initialize the classifier
        model = ImageClassifier(modelArch=model_arch)

        model.train(trainDataDir='data/buildings_training_set', nepochs=200)


def image_loader(img_transforms, image_name):
    image = Image.open(image_name).convert("RGB")
    image = img_transforms(image).float()
    image = image.unsqueeze(0)  
    return image.to(torch.device('cpu'))


def evaluate_models(model_architectures):
    for model in model_architectures:
        evaluate_model(model)


def evaluate_model(model_name):
    model = ImageClassifier()
    model = torch.load(f'tmp/models/{model_name}.pth', map_location=torch.device('cpu'))
    model.eval()

    data_dir = Path('data/buildings_eval_set')

    classes= sorted(['metal', 'not_metal', 'no_bldg'])

    times_per_case = []
    imgs_per_case = []

    all_true = []
    all_preds = []

    img_transforms = transforms.Compose([
            transforms.Resize(384),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    for imgdir in data_dir.iterdir():
        t0 = time.time()

        case = imgdir.name
        case_images = [p for p in imgdir.iterdir()]

        # same for all case images
        truth = get_real_answer(case)
        # We skip some cases
        if not truth:
            continue
        
        for img in case_images:

            image = image_loader(img_transforms, img)
            # Forward pass
            _, pred = torch.max(model(image),1)
            pred = classes[pred]
            all_true.append(truth)
            all_preds.append(pred)


        print(f'Case {case} done')

    try:
        accuracy = metrics.accuracy_score(all_true, all_preds)
        precision_micro = metrics.precision_score(all_true, all_preds, average='micro')
        precision_macro = metrics.precision_score(all_true, all_preds, average='macro')
        recall_micro = metrics.recall_score(all_true, all_preds, average = 'micro')
        recall_macro = metrics.recall_score(all_true, all_preds, average = 'macro')
        f1_micro = metrics.f1_score(all_true, all_preds, average = 'micro')
        f1_macro = metrics.f1_score(all_true, all_preds, average = 'macro')

    except:
        import IPython
        IPython.embed()


    time_taken = round(time.time() - t0, 2)
    times_per_case.append(time_taken)
    imgs_per_case.append(len(case_images))

    print(
        f"{round(sum(times_per_case) / len(times_per_case), 2)} s/case for {round(sum(imgs_per_case) / len(imgs_per_case), 2)} image/case on avg"
    )


    RESULTS_DIR = 'output'
    # Compute statistics and write out the results
    with open(
        f"{RESULTS_DIR}/{model_name}_results.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as outfile:

        csvwriter = csv.writer(outfile)

        header = [
            "Accuracy",
            "Recall_micro",
            "Recall_macro",
            "Precision_micro",
            "Precision_macro",
            'F1_micro',
            'F1_macro',
        ]

        csvwriter.writerow(header)

        csvwriter.writerow(
            [
                accuracy,
                precision_micro,
                precision_macro,
                recall_micro,
                recall_macro,
                f1_micro,
                f1_macro,
            ]
        )

        csvwriter.writerow([""])

if __name__ == '__main__':
    # BRAILS supports more models, these are just the ones I could realistically train on my laptop
    # The bigger ones would take a few days to train.
    model_architectures = ["convnextb", "convnexts", "efficientnetv2m", "efficientnetv2s"]
    train_models(model_architectures)
    evaluate_models(model_architectures)
