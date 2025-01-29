import torch
from tqdm import tqdm
import os
import glob
import pandas as pd
from PIL import Image
from transformers import AutoImageProcessor, BertTokenizer
import config.config as cfg

# 创建一个Dataset类,用于处理图片数据, 使其符合ConvNeXT模型的输入要求, 该类继承自torch.utils.data.Dataset
class MultiModel_Dataset(torch.utils.data.Dataset):
    # 构造函数, 传入图片文件夹的路径
    def __init__(self, image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file):
        self.images = PreProcessDataset(image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file)

    # 实现__len__方法, 返回数据集的大小
    def __len__(self):
        return len(self.images)

    # 实现__getitem__方法, 传入索引, 返回对应索引的数据
    def __getitem__(self, idx):
        return self.images[idx]

# 创建DataLoader, 用于加载上述的DataSet
class MultiModel_DataLoader():
    def __init__(self, dataset, batch_size, shuffle):
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

def PreProcessDataset(image_root_folder, report_csv_file, answer1_csv_file, answer2_csv_file, answer3_csv_file):
    preprocessor = AutoImageProcessor.from_pretrained(cfg.ConvNeXT_NAME)

    # 打开report_csv_file文件, 读取其中的内容
    reports_csv_file = pd.read_csv(report_csv_file)

    # 打开answer1_csv_file文件, 读取其中的内容
    answer1_csv_file = pd.read_csv(answer1_csv_file)

    # 打开answer2_csv_file文件, 读取其中的内容
    answer2_csv_file = pd.read_csv(answer2_csv_file)

    # 打开answer3_csv_file文件, 读取其中的内容
    answer3_csv_file = pd.read_csv(answer3_csv_file)

    # 打开image_root_folder文件夹，获取该文件夹内的所有子文件夹名称，并保存至patient_image_folders内。
    patient_image_folders = os.listdir(image_root_folder)

    datas = []

    for patient_image_folder in tqdm(patient_image_folders):
        # 读取&处理image
        # 打开patient_image_folder文件夹，获取该文件夹内的所有以jpg结尾的图像文件的名称，并保存至patient_image_files内。
        patient_images = glob.glob(os.path.join(image_root_folder, patient_image_folder, '*.jpg'))
        patient_images.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))

        images = []
        for patient_image in patient_images:
            image = Image.open(patient_image)
            image = image.convert('RGB')
            image = preprocessor(image, return_tensors="pt").pixel_values.squeeze()
            images.append(image)
        images = torch.stack(images)[:len(images)]

        # 读取&处理report
        report = PreProcessReport(reports_csv_file, patient_image_folder)

        # 读取&处理answer
        answers = PreProcessAnswer(answer1_csv_file, answer2_csv_file, answer3_csv_file, patient_image_folder)

        valid_img_num_file = os.path.join(image_root_folder, patient_image_folder, 'valid_img_num.txt')
        with open(valid_img_num_file, 'r') as f:
            valid_img_num = int(f.read())

        datas.append({"images": images, "report": report, "answers": answers, "valid_image_num": valid_img_num})

    return datas

def PreProcessReport(report_csv_file, patient_id):
    tokenizer = BertTokenizer.from_pretrained(cfg.Bert_NAME)

    # 在reports中的第一列中寻找与patient_image_folder相同的行，将其对应的第二列内容作为报告
    report = report_csv_file[report_csv_file.iloc[:, 0] == patient_id].iloc[:, 1].values[0]

    report_max_len = cfg.Report_Max_Length
    question_max_len = cfg.Question_Max_Length
    answer_max_len = cfg.Answer_Max_Length
    pad_len = cfg.Pad_Length

    # 读取text_csv_file的第二列作为报告
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

    # report中有些word在str的len的计算中的结果与tokenized_report的shape不一致, 所以这里使用tokenized_report.shape[0]来计算report的有效长度
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
    # 在answer1_csv_file中的第一列中寻找与patient_id相同的行，将该行的从第2列开始的所有内容作为answer1
    answer1 = torch.tensor(answer1_csv_file[answer1_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)
    answer2 = torch.tensor(answer2_csv_file[answer2_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)
    answer3 = torch.tensor(answer3_csv_file[answer3_csv_file.iloc[:, 0] == patient_id].iloc[:, 1:].values[0], dtype=torch.long)

    answer_obj = {"answer1": answer1, "answer2": answer2, "answer3": answer3}

    return answer_obj

if __name__ == "__main__":
    dataset = MultiModel_Dataset(cfg.Train_Image_Root_Folder, cfg.Train_Report_CSV_File, cfg.Train_Answer1_CSV_File, cfg.Train_Answer2_CSV_File, cfg.Train_Answer3_CSV_File)
    dataloader = MultiModel_DataLoader(dataset, cfg.Patient_Batch_Size, shuffle=True)
    print(dataset)
