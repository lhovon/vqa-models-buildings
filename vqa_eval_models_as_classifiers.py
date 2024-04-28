import csv
import time
import torch
import IPython
import numpy as np

from PIL import Image
from pathlib import Path
from sklearn import metrics
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


# The empirically measured sensitivity and specificity
# of each question. Obtained by runing the function
# evaluate_model_on_known_cases()
Q_SENSI_SPECI = {
    "vilt-b32-finetuned-vqa": [
        (0.815699659, 0.933333333),
        (0.960182025, 0.705263158),
        (0.956769056, 0.743859649),
        (0.926052332, 0.810526316),
        (0.908536585, 0.573),
        (0.914634146, 0.552),
        (0.676829268, 0.889),
        (0.12195122, 0.995),
    ],
    "blip-vqa-capfilt-large" : [
        (0.970420933, 0.61754386),
        (0.970420933, 0.628070175),
        (0.969283276, 0.621052632),
        (0.976109215, 0.568421053),
        (0.823170732, 0.658),
        (0.81097561, 0.693),
        (0.682926829, 0.845),
        (0.670731707, 0.952),
    ]
}

# Baseline rates of cases with a building
# and with metal buildings. Empirically
# estimated by sampling the dataset
PREVALENCE_BLDGS = 0.6509433962264151
PREVALENCE_METAL_BLDGS = 0.11320754716981132

PROB_THRESHOLD = 0.5


def probability_of_answers_given_positive_test(model_name, answers):
    """
    Calculate P(answers | test) using the answers and sensitivities
    Assuming independence of the questions:
    P(answers | test) = P(q1 = ans1 | test) * P(q2 = ans2 | test) * ...

    If the answer is positive then 
        P(q = positive answer | test ) = sensitivity of q
    else 
        P(q = negative answer | test ) = (1 - sensitivity of q)
    """
    p_answers_given_bldg = 1
    p_answers_given_metal = 1

    assert len(answers) == len(QUESTIONS)

    for i, ans in enumerate(answers):
        q_sensitivity = Q_SENSI_SPECI[model_name][i][0]

        if i < 4:
            if ans == 'yes':
                p_answers_given_bldg *= q_sensitivity
            else:
                p_answers_given_bldg *= (1 - q_sensitivity)
        else:
            if ans in ['yes', 'metal']:
                p_answers_given_metal *= q_sensitivity
            else:
                p_answers_given_metal *= (1 - q_sensitivity)

    return p_answers_given_bldg, p_answers_given_metal


def probability_of_answers_given_negative_test(model_name, answers):

    """
    Calculate P(answers | !test ) using the answers and sensitivities
    Assuming independence of the questions:
    P(answers | !test) = P(q1 = ans1 | not !test) * P(q2 = ans2 | not !test) * ...

    If the answer is negative then 
        P(q = negative answer | !test ) = SPECIFICITY of q
    else 
        P(q = positive answer | !test ) = (1 - specificity of q)
    """
    p_answers_given_not_bldg = 1
    p_answers_given_not_metal = 1

    assert len(answers) == len(QUESTIONS)

    for i, ans in enumerate(answers):
        q_specificity = Q_SENSI_SPECI[model_name][i][1]

        if i < 4:
            if ans == 'yes':
                p_answers_given_not_bldg *= (1 - q_specificity) 
            else:
                p_answers_given_not_bldg *= q_specificity
        else:
            if ans in ['yes', 'metal']:
                p_answers_given_not_metal *= (1 - q_specificity)
            else:
                p_answers_given_not_metal *= q_specificity

    return p_answers_given_not_bldg, p_answers_given_not_metal


def compute_probabilities(model_name, answers_per_image):
    """
    Computes the probability that a given case contains a building, 
    and that it contains a metal buidling, based on the provided answers.

    Takes in an array of arrays containing each question's answer for each image
    all_answers = [
        [answers for image 1...],
        [answers for image 2...],
        ...
    ]
    """
    all_p_bldgs = []
    all_p_metals = []

    for answer_to_each_question in answers_per_image:

        P_answers_given_bldg, P_answers_given_metal = probability_of_answers_given_positive_test(model_name, answer_to_each_question)
        P_answers_given_not_bldg, P_answers_given_not_metal = probability_of_answers_given_negative_test(model_name, answer_to_each_question)

        P_answers_bldg = P_answers_given_bldg * PREVALENCE_BLDGS + P_answers_given_not_bldg * (1 - PREVALENCE_BLDGS)
        P_bldg_given_answers = (P_answers_given_bldg * PREVALENCE_BLDGS) / P_answers_bldg
        all_p_bldgs.append(P_bldg_given_answers)

        P_answers_metal = P_answers_given_metal * PREVALENCE_METAL_BLDGS + P_answers_given_not_metal * (1 - PREVALENCE_METAL_BLDGS)
        P_metal_given_answers = (P_answers_given_metal * PREVALENCE_METAL_BLDGS) / P_answers_metal
        all_p_metals.append(P_metal_given_answers)

    # Average the answers
    avg_p_bldg = np.mean(all_p_bldgs)
    avg_p_metal = np.mean(all_p_metals)

    return round(avg_p_bldg, 4), round(avg_p_metal, 4)



def get_predicted_class(p_bldg, p_metal):

    if p_bldg < PROB_THRESHOLD and p_metal < PROB_THRESHOLD:
        return 'no_bldg'
    
    if p_bldg >= PROB_THRESHOLD and p_metal < PROB_THRESHOLD:
        return 'not_metal'
    
    if p_metal >= PROB_THRESHOLD:
        return "metal"
     

def run_model_with_probabilities(model_name):
    global model
    global processor
    global model_forward_fn

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    DATADIR = Path("data/buildings_eval_set")

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

    all_true = []
    all_preds = []

    for imgdir in DATADIR.iterdir():
        t0 = time.time()

        case = imgdir.name
        ground_truth = get_real_answer(case)
        if not ground_truth:
            continue
        images = [Image.open(p) for p in imgdir.iterdir()]
        if not images:
            continue
        answers = model_forward_fn(images, QUESTIONS)
        answers_per_image = np.transpose(answers)

        p_bldg, p_metal = compute_probabilities(model_name, answers_per_image)

        pred = get_predicted_class(p_bldg, p_metal)
        

        all_true.append(ground_truth)
        all_preds.append(pred)

        time_taken = round(time.time() - t0, 2)
        print(
            f"{case} - answered {len(QUESTIONS)} questions on {len(images)} images in {time_taken}s"
        )

    try:

        accuracy = metrics.accuracy_score(all_true, all_preds)
        recall_micro = metrics.recall_score(all_true, all_preds, average = 'micro')
        recall_macro = metrics.recall_score(all_true, all_preds, average = 'macro')
        precision_micro = metrics.precision_score(all_true, all_preds, average='micro')
        precision_macro = metrics.precision_score(all_true, all_preds, average='macro')
        f1_micro = metrics.f1_score(all_true, all_preds, average = 'micro')
        f1_macro = metrics.f1_score(all_true, all_preds, average = 'macro')

    except:
            
        import traceback
        traceback.print_exc()
        import IPython
        IPython.embed()


    # Compute statistics and write out the results
    with open(
        f"{RESULTS_DIR}/{model_name}_classification_results.csv",
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


if __name__ == "__main__":

    for model_name in [
        "Salesforce/blip-vqa-capfilt-large",
        "dandelin/vilt-b32-finetuned-vqa",
    ]:
        run_model_with_probabilities(model_name=model_name)
