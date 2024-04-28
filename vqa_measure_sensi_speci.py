import csv
import time
import torch
import IPython

from PIL import Image
from pathlib import Path
from collections import Counter
from transformers import ViltProcessor, ViltForQuestionAnswering
from transformers import BlipForQuestionAnswering, BlipProcessor
from known_cases import CASES_SET_NO_BLDG, CASES_SET_METAL, CASES_SET_NOT_METAL

RESULTS_DIR = Path('output')

QUESTIONS = [
    "Is a building clearly visible?",
    "Is there a building in the image?",
    "Is a building present in the image?",
    "Can you clearly see a building in the image?",
    "Is there a metal prefabricated building in the image?",
    "Is there a metal prefabricated building?",
    "Is there a metal building?",
    "If there is a building, what is it made of?",
]

def answer_questions_ViLT(images, questions):
    """
    Returns a list of lists of dimensions n_questions x n_images
    """
    n_images = len(images)

    with torch.no_grad():
        images = processor.image_processor(images, return_tensors="pt")

        # tokenize questions
        # list of dicts with input_ids, token_type_ids, attention_mask
        questions = [
            processor.tokenizer(text=q, return_tensors="pt") for q in questions
        ]

        answers_per_question = []

        # Iterate over the questions and collect answers
        # We separately pose each question to all images
        for question in questions:

            # Model expects the questions to be in a batch of the same size as the image batch
            # even though it's the same question repeated multiple times
            for k, v in question.items():
                question[k] = v.expand([n_images, v.shape[1]])

            model_input = dict(images, **question)

            outputs = model(**model_input)
            logits = outputs.logits

            image_answers = [
                model.config.id2label[i] for i in logits.argmax(-1).tolist()
            ]
            answers_per_question.append(image_answers)

    return answers_per_question


def answer_questions_BLIP(images, questions):
    """
    Answers all questions on all images.
    Questions are answered one by one on the image batch.
    Taken from https://github.com/salesforce/BLIP/issues/78#issuecomment-1463388205
    """
    # Check for None's in input, dynamicaly change
    # the batch size to number of non null elements
    if isinstance(images, list):
        images = list(filter(lambda img: img is not None, images))
        batch_size = len(images)
    else:
        batch_size = 1

    # first element are the answers to the first question
    answers_to_questions = []

    # preprocess image
    images = processor.image_processor(images, return_tensors="pt")

    # tokenize texts
    questions = [processor.tokenizer(text=q, return_tensors="pt") for q in questions]

    with torch.no_grad():
        # compute image embedding
        vision_outputs = model.vision_model(pixel_values=images["pixel_values"])
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long)

        for question in questions:
            # compute text encodings
            question_outputs = model.text_encoder(
                input_ids=torch.cat(tuple([question["input_ids"]] * batch_size)),
                attention_mask=None,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=False,
            )
            question_embeds = question_outputs[0]
            question_attention_mask = torch.ones(
                question_embeds.size()[:-1], dtype=torch.long
            )
            bos_ids = torch.full(
                (question_embeds.size(0), 1), fill_value=model.decoder_start_token_id
            )

            outputs = model.text_decoder.generate(
                input_ids=bos_ids,
                eos_token_id=model.config.text_config.sep_token_id,
                pad_token_id=model.config.text_config.pad_token_id,
                encoder_hidden_states=question_embeds,
                encoder_attention_mask=question_attention_mask,
                max_new_tokens=3,
            )

            # BLIP answers one question at a time for all images
            # Here we take the answer of the ith image and append it to the img answers
            if batch_size > 1:
                ans_per_img = []
                for out in outputs:
                    answer = processor.decode(out, skip_special_tokens=True)
                    ans_per_img.append(answer)
                answers_to_questions.append(ans_per_img)
            else:
                answer = processor.decode(outputs[0], skip_special_tokens=True)
                answers_to_questions.append(answer)

        return answers_to_questions


def get_real_answer(case):

    if case in CASES_SET_NO_BLDG:
        return "no_bldg"

    if case in CASES_SET_METAL:
        return "prefab_metal"

    if case in CASES_SET_NOT_METAL:
        return "not_metal"


def verify_answer(q, ans, actual):
    if q in [
        "Is a building clearly visible?",
        "Is there a building in the image?",
        "Is a building present in the image?",
        "Can you clearly see a building in the image?",
    ]:
        if actual == "no_bldg":
            if ans == "no":
                return {"tn": 1}
            else:
                return {"fp": 1}
        # in every other case, there's a bldg
        else:
            if ans == "yes":
                return {"tp": 1}
            else:
                return {"fn": 1}

    if q in [
        "Is there a metal prefabricated building in the image?",
        "Is there a metal prefabricated building?",
    ]:
        if actual == "prefab_metal":
            if ans == "yes":
                return {"tp": 1}
            else:
                return {"fn": 1}
        else:
            if ans == "no":
                return {"tn": 1}
            else:
                return {"fp": 1}

    if q == "Is there a metal building?":
        if actual in ["prefab_metal", "metal_not_prefab"]:
            if ans == "yes":
                return {"tp": 1}
            else:
                return {"fn": 1}
        # in every other case, there is no metal prefab bldg in the img
        else:
            if ans == "no":
                return {"tn": 1}
            else:
                return {"fp": 1}

    if q == "If there is a building, what is it made of?":
        if actual in ["prefab_metal", "metal_not_prefab"]:
            if ans == "metal":
                return {"tp": 1}
            else:
                return {"fn": 1}
        else:
            if ans == "metal":
                return {"fp": 1}
            else:
                return {"tn": 1}



def run_experiments(model_name):

    global model
    global processor
    global model_forward_fn

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Evaluating {model_name} on known cases")
    if "blip" in model_name:
        model = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-capfilt-large"
        )
        processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        model_forward_fn = answer_questions_BLIP
    elif "vilt" in model_name:
        processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa"
        )
        model_forward_fn = answer_questions_ViLT
    else:
        raise Exception("Unsupported model")

    model_name = model_name.split("/")[1]

    data_dir = Path("data/buildings_eval_set")

    counters = {q: Counter() for q in QUESTIONS}

    times_per_case = []
    imgs_per_case = []

    for imgdir in data_dir.iterdir():
        t0 = time.time()

        case = imgdir.name

        images = [Image.open(p) for p in imgdir.iterdir()]
        if not images:
            continue
        answers = model_forward_fn(images, QUESTIONS)

        ground_truth = get_real_answer(case)

        # For each question, tally up results
        for i, q in enumerate(QUESTIONS):
            ans_per_img = answers[i]

            for ans in ans_per_img:
                counters[q].update(verify_answer(q, ans, ground_truth))

        time_taken = round(time.time() - t0, 2)
        times_per_case.append(time_taken)
        imgs_per_case.append(len(images))
        print(
            f"Answered {len(QUESTIONS)} questions on {len(images)} images in {time_taken}s"
        )


    print(
        f"{round(sum(times_per_case) / len(times_per_case), 2)} s/case for {round(sum(imgs_per_case) / len(imgs_per_case), 2)} image/case on avg"
    )


    # Compute statistics and write out the results
    with open(
        f"{RESULTS_DIR}/{model_name}_results.csv",
        "w",
        encoding="utf-8",
        newline="",
    ) as outfile:

        csvwriter = csv.writer(outfile)

        header = [
            "Question",
            "TP",   # True Positives
            "FP",   # False Positives
            "TN",   # True Negatives
            "FN",   # False Negatives
            "Sensitivity", # == Recall
            "Specificity",
            "FPR",  # False Positive Rate
            "FNR",
            "PPV",  # Positive Predictive Value
            "NPV",  # Negative Predictive Value
            "1 - NPV",
            "Accuracy",
            "Precision",
            "f1"
        ]

        csvwriter.writerow(header)

        for question, counter in counters.items():
            # All question scores summed up accross all panoramas
            P = counter["tp"] + counter["fn"]
            N = counter["fp"] + counter["tn"]

            sensitivity = counter["tp"] / P if P != 0 else "n/a"
            specificity = counter["tn"] / N if N != 0 else "n/a"
            FNR = counter["fn"] / P if P != 0 else "n/a"
            FPR = counter["fp"] / N if N != 0 else "n/a"
            PPV = counter["tp"] / (counter["tp"] + counter["fp"])
            NPV = counter["tn"] / (counter["tn"] + counter["fn"])
            accuracy = (counter["tp"] + counter["tn"]) / (counter["tp"] + counter["fn"] + counter["fp"] + counter["tn"])
            precision = counter["tp"] / (counter["tp"] + counter["fp"])
            f1 = 2 * precision * sensitivity / (precision + sensitivity)

            csvwriter.writerow(
                [
                    question,
                    counter["tp"],
                    counter["fp"],
                    counter["tn"],
                    counter["fn"],
                    sensitivity,
                    specificity,
                    FPR,
                    FNR,
                    PPV,
                    NPV,
                    1 - NPV,
                    accuracy,
                    precision,
                    f1
                ]
            )

        csvwriter.writerow([""])


if __name__ == "__main__":

    for model_name in [
        "Salesforce/blip-vqa-capfilt-large",
        "dandelin/vilt-b32-finetuned-vqa",
    ]:
        run_experiments(model_name=model_name)
