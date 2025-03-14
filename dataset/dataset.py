import torch
from tqdm import tqdm
import os
import glob
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, BertTokenizer
import config.config as cfg

class MultiModel_Dataset(torch.utils.data.Dataset):
    def __init__(self, image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file):
        self.images = PreProcessDataset(image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

class MultiModel_DataLoader():
    def __init__(self, dataset, batch_size, shuffle):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def PreProcessDataset(image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file):
    preprocessor = AutoImageProcessor.from_pretrained(cfg.ConvNeXT_NAME)

    reports_csv_file = pd.read_csv(report_csv_file)

    answer1_csv_file = pd.read_csv(answer1_csv_file)

    answer2_csv_file = pd.read_csv(answer2_csv_file)

    answer3_csv_file = pd.read_csv(answer3_csv_file)

    patient_image_folders = os.listdir(image_root_folder)

    datas = []

    for patient_image_folder in tqdm(patient_image_folders):
        patient_images = glob.glob(os.path.join(image_root_folder, patient_image_folder, '*.jpg'))
        patient_images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        images = []
        for patient_image in patient_images:
            image = Image.open(patient_image)
            image = image.convert('RGB')
            image = preprocessor(image, return_tensors="pt").pixel_values.squeeze()
            images.append(image)
        images = torch.stack(images)[:len(images)]

        report = PreProcessReport(reports_csv_file, patient_image_folder)

        answers = PreProcessAnswer(answer1_csv_file, answer2_csv_file, answer3_csv_file, patient_image_folder)

        valid_img_num_file = os.path.join(image_root_folder, patient_image_folder, 'valid_img_num.txt')
        with open(valid_img_num_file, 'r') as f:
            valid_img_num = int(f.read())

        datas.append({"images": images, "report": report, "answers": answers, "valid_image_num": valid_img_num})

    return datas

def PreProcessReport(report_csv_file, patient_id):
    tokenizer = BertTokenizer.from_pretrained(cfg.Bert_NAME)

    report = report_csv_file[report_csv_file.iloc[:, 0] == patient_id].iloc[:, 1].values[0]

    report_max_len = cfg.Report_Max_Length
    question_max_len = cfg.Question_Max_Length
    answer_max_len = cfg.Answer_Max_Length
    pad_len = cfg.Pad_Length

    question1 = "问题：脑内多发哪些病灶，或无病灶"
    question2 = "问题：是否存在脑白质脱髓鞘？"
    question3 = "问题：对明显的异常表现进行详细描述。"

    tokenized_report = tokenizer(report, return_tensors='pt', add_special_tokens=False)['input_ids'].squeeze()
    question1_seq = tokenizer.sep_token + question1
    question2_seq = tokenizer.sep_token + question2
    question3_seq = tokenizer.sep_token + question3
    answer_seq = tokenizer.sep_token + "回答：" + tokenizer.mask_token

    token_type_ids = ([0] * (report_max_len + 1) +
                      [0] * (question_max_len + 1) + [1] * (answer_max_len + 1) +
                      [0] * (question_max_len + 1) + [1] * (answer_max_len + 1) +
                      [0] * (question_max_len + 1) + [1] * (answer_max_len + 1) +
                      [0] * pad_len)
    token_type_ids = torch.tensor(token_type_ids)

    report_valid_len = tokenized_report.shape[0]
    question1_valid_len = len(question1_seq) - 5
    question2_valid_len = len(question2_seq) - 5
    question3_valid_len = len(question3_seq) - 5

    attention_mask = ([1] * (report_valid_len + 1) + [0] * (report_max_len - report_valid_len) +
                      [1] * (question1_valid_len + 1) + [0] * (question_max_len - question1_valid_len) + [1] * (answer_max_len + 1) +
                      [1] * (question2_valid_len + 1) + [0] * (question_max_len - question2_valid_len) + [1] * (answer_max_len + 1) +
                      [1] * (question3_valid_len + 1) + [0] * (question_max_len - question3_valid_len) + [1] * (answer_max_len + 1) +
                      [0] * (pad_len))
    attention_mask = torch.tensor(attention_mask)

    report += (report_max_len - report_valid_len) * tokenizer.pad_token
    question1_seq += (question_max_len - question1_valid_len) * tokenizer.pad_token
    question2_seq += (question_max_len - question2_valid_len) * tokenizer.pad_token
    question3_seq += (question_max_len - question3_valid_len) * tokenizer.pad_token

    full_text = (report +
                question1_seq + answer_seq +
                question2_seq + answer_seq +
                question3_seq + answer_seq)

    input_ids = tokenizer(full_text, return_tensors='pt', max_length=cfg.Bert_Max_Length, truncation=True, padding='max_length')['input_ids'].squeeze()

    report_obj = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}

    return report_obj

def PreProcessAnswer(answer1_csv_file, answer2_csv_file, answer3_csv_file, patient_id):
    answer1 = torch.tensor(answer1_csv_file[answer1_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)
    answer2 = torch.tensor(answer2_csv_file[answer2_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)
    answer3 = torch.tensor(answer3_csv_file[answer3_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)

    answer_obj = {"answer1": answer1, "answer2": answer2, "answer3": answer3}

    return answer_obj

if __name__ == "__main__":
    dataset = MultiModel_Dataset(cfg.Train_Image_Root_Folder, cfg.Train_Report_CSV_File, cfg.Train_Answer1_CSV_File, cfg.Train_Answer2_CSV_File, cfg.Train_Answer3_CSV_File)
    dataloader = MultiModel_DataLoader(dataset, cfg.Patient_Batch_Size, shuffle=True)
    print(dataset)
