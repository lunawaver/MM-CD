import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm

import config.config as cfg

from model.MMCD import MMCD

from dataset.dataset import MultiModel_Dataset, MultiModel_DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings

warnings.filterwarnings("ignore")

def get_train_data_and_test_dataloader():
    train_image_root_folder = cfg.Train_Image_Root_Folder
    test_image_root_folder = cfg.Test_Image_Root_Folder

    train_report_csv_file = cfg.Train_Report_CSV_File
    test_report_csv_file = cfg.Test_Report_CSV_File

    train_answer1_csv_file = cfg.Train_Answer1_CSV_File
    train_answer2_csv_file = cfg.Train_Answer2_CSV_File
    train_answer3_csv_file = cfg.Train_Answer3_CSV_File
    test_answer1_csv_file = cfg.Test_Answer1_CSV_File
    test_answer2_csv_file = cfg.Test_Answer2_CSV_File
    test_answer3_csv_file = cfg.Test_Answer3_CSV_File

    patient_batch_size = cfg.Patient_Batch_Size

    train_dataset = MultiModel_Dataset(train_image_root_folder, train_report_csv_file, train_answer1_csv_file, train_answer2_csv_file, train_answer3_csv_file)
    train_dataloader = MultiModel_DataLoader(train_dataset, batch_size=patient_batch_size, shuffle=False)

    test_dataset = MultiModel_Dataset(test_image_root_folder, test_report_csv_file, test_answer1_csv_file, test_answer2_csv_file, test_answer3_csv_file)
    test_dataloader = MultiModel_DataLoader(test_dataset, batch_size=patient_batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def run(model, train_dataloader, test_dataloader):
    device = cfg.device
    epoch_num = cfg.epoch_num

    best_f1_score = 0

    optimizer = torch.optim.AdamW([
        {'params': [p for n, p in list(model.image_encoder.named_parameters())], 'lr': 1e-4},
        {'params': [p for n, p in list(model.multi_image_weight_generator_model.named_parameters())], 'lr': 1e-4},
        {'params': [p for n, p in list(model.multi_image_fusion_model.named_parameters())], 'lr': 1e-4},
        {'params': [p for n, p in list(model.report_encoder.named_parameters())], 'lr': 5e-5},

        {'params': [p for n, p in list(model.multi_model_fusion_model.named_parameters())], 'lr': 1e-4},

        {'params': [p for n, p in list(model.pred_answer1_none_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_lacunar_infarction_lesion_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_ischemic_lesion_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_softening_lesion_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_lacunar_lesion_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_infarction_lesion_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer1_hemorrhagic_lesion_layer.named_parameters())], 'lr': 5e-5},

        {'params': [p for n, p in list(model.pred_answer2_layer.named_parameters())], 'lr': 5e-5},

        {'params': [p for n, p in list(model.pred_answer3_none_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer3_fptoi_lobe_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer3_basal_ganglia_and_corona_radiata_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer3_cerebral_hemisphere_and_cerebellar_hemisphere_layer.named_parameters())], 'lr': 5e-5},
        {'params': [p for n, p in list(model.pred_answer3_other_layer.named_parameters())], 'lr': 5e-5}
    ], betas=(0.9, 0.99), weight_decay=0.01)

    lr_decay = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    criterion = nn.CrossEntropyLoss().to(cfg.device)

    for epoch in range(epoch_num):
        model.train()
        
        answer1_pred_array_for_eval = [[], [], [], [], [], [], []]
        answer2_pred_array_for_eval = []
        answer3_pred_array_for_eval = [[], [], [], [], []]

        answer1_gt_array = [[], [], [], [], [], [], []]
        answer2_gt_array = []
        answer3_gt_array = [[], [], [], [], []]

        print("epoch: ", epoch)
        total_loss = 0
        total_answer1_loss = 0
        total_answer2_loss = 0
        total_answer3_loss = 0
        count = 0
        accumulation_step = 32
        for batch in tqdm(train_dataloader):

            images = batch["images"].squeeze().to(device)

            input_ids = batch["report"]["input_ids"].to(device)
            token_type_ids = batch["report"]["token_type_ids"].to(device)
            attention_mask = batch["report"]["attention_mask"].to(device)

            gt_answer1 = batch["answers"]["answer1"].to(device)
            gt_answer2 = batch["answers"]["answer2"].to(device)
            gt_answer3 = batch["answers"]["answer3"].to(device)

            valid_image_num = batch["valid_image_num"].to(device)

            pred_answer1_list, pred_answer2, pred_answer3_list = model.forward(images, input_ids, token_type_ids, attention_mask, valid_image_num, is_train=True)

            answer1_loss = 0
            answer2_loss = 0
            answer3_loss = 0

            for idx, pred in enumerate(pred_answer1_list):
                _, pred_class = torch.max(pred, 1)
                gt_class = gt_answer1[0][idx]
                answer1_loss += criterion(pred_answer1_list[idx], torch.tensor([gt_answer1[0, idx].item()], device=device))
                answer1_pred_array_for_eval[idx].append(pred_class.item())
                answer1_gt_array[idx].append(gt_class.item())
                
            _, pred_class = torch.max(pred_answer2, 1)
            answer2_gt_class = gt_answer2[0]
            answer2_loss += criterion(pred_answer2, answer2_gt_class)
            answer2_pred_array_for_eval.append(pred_class.item())
            answer2_gt_array.append(answer2_gt_class.item())

            for idx, pred in enumerate(pred_answer3_list):
                _, pred_class = torch.max(pred, 1)
                gt_class = gt_answer3[0][idx]
                answer3_loss += criterion(pred_answer3_list[idx], torch.tensor([gt_answer3[0, idx].item()], device=device))
                answer3_pred_array_for_eval[idx].append(pred_class.item())
                answer3_gt_array[idx].append(gt_class.item())

            loss = answer1_loss + answer2_loss + answer3_loss
            loss = loss / accumulation_step
            loss.backward()
            count += 1
            total_loss += loss.detach().item()

            if count == accumulation_step:
                optimizer.step()
                optimizer.zero_grad()
                count = 0

            total_answer1_loss += answer1_loss.item()
            total_answer2_loss += answer2_loss.item()
            total_answer3_loss += answer3_loss.item()
        
        lr_decay.step()

        loss = total_loss / len(train_dataloader) * accumulation_step
        with open(cfg.Model_Save_Dir + "train_log.txt", "a") as f:
            f.write("epoch " + str(epoch + 1) + " loss: " + str(loss) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", answer1_loss: " + str(total_answer1_loss / len(train_dataloader)) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", answer2_loss: " + str(total_answer2_loss / len(train_dataloader)) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", answer3_loss: " + str(total_answer3_loss / len(train_dataloader)) + "\n\n")

        answer1_label_names = ["无病灶", "腔梗灶", "缺血灶", "软化灶", "腔隙灶", "梗死灶", "出血灶"]
        answer3_label_names = ["无异常表现", "额顶颞枕岛叶", "基底节放射冠", "大脑半球或小脑半球", "其他"]

        for idx, label in enumerate(answer1_label_names):
            answer1_pred_target_label_array = answer1_pred_array_for_eval[idx]
            answer1_gt_target_label_array = answer1_gt_array[idx]
            label_acc_score = accuracy_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_precision_score = precision_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_recall_score = recall_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_f1_score = f1_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            with open(cfg.Model_Save_Dir + "train_log.txt", "a") as f:
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", acc: " + str(label_acc_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", precision: " + str(label_precision_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", recall: " + str(label_recall_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", f1_score: " + str(label_f1_score) + "\n\n")

        answer2_acc_score = accuracy_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_precision_score = precision_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_recall_score = recall_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_f1_score = f1_score(answer2_gt_array, answer2_pred_array_for_eval)
        with open(cfg.Model_Save_Dir + "train_log.txt", "a") as f:
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", acc: " + str(answer2_acc_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", precision: " + str(answer2_precision_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", recall: " + str(answer2_recall_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", f1_score: " + str(answer2_f1_score) + "\n\n")

        for idx, label in enumerate(answer3_label_names):
            answer3_pred_target_label_array = answer3_pred_array_for_eval[idx]
            answer3_gt_target_label_array = answer3_gt_array[idx]
            label_acc_score = accuracy_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_precision_score = precision_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_recall_score = recall_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_f1_score = f1_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            with open(cfg.Model_Save_Dir + "train_log.txt", "a") as f:
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", acc: " + str(label_acc_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", precision: " + str(label_precision_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", recall: " + str(label_recall_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", f1_score: " + str(label_f1_score) + "\n\n")
        
        model.eval()
        answer1_pred_array_for_eval = [[], [], [], [], [], [], []]
        answer2_pred_array_for_eval = []
        answer3_pred_array_for_eval = [[], [], [], [], []]

        answer1_gt_array = [[], [], [], [], [], [], []]
        answer2_gt_array = []
        answer3_gt_array = [[], [], [], [], []]
        
        total_f1_score = 0
        
        for batch in tqdm(test_dataloader):

            images = batch["images"].squeeze().to(device)

            input_ids = batch["report"]["input_ids"].to(device)
            token_type_ids = batch["report"]["token_type_ids"].to(device)
            attention_mask = batch["report"]["attention_mask"].to(device)

            gt_answer1 = batch["answers"]["answer1"].to(device)
            gt_answer2 = batch["answers"]["answer2"].to(device)
            gt_answer3 = batch["answers"]["answer3"].to(device)

            valid_image_num = batch["valid_image_num"].to(device)

            with torch.no_grad():
                pred_answer1_list, pred_answer2, pred_answer3_list = model.forward(images, input_ids, token_type_ids, attention_mask, valid_image_num, is_train=False)
            
            for idx, pred in enumerate(pred_answer1_list):
                _, pred_class = torch.max(pred, 1)
                gt_class = gt_answer1[0][idx]
                answer1_pred_array_for_eval[idx].append(pred_class.item())
                answer1_gt_array[idx].append(gt_class.item())

            _, pred_class = torch.max(pred_answer2, 1)
            answer2_gt_class = gt_answer2[0]
            answer2_pred_array_for_eval.append(pred_class.item())
            answer2_gt_array.append(answer2_gt_class.item())

            for idx, pred in enumerate(pred_answer3_list):
                _, pred_class = torch.max(pred, 1)
                gt_class = gt_answer3[0][idx]
                answer3_loss += criterion(pred_answer3_list[idx], torch.tensor([gt_answer3[0, idx].item()], device=device))
                answer3_pred_array_for_eval[idx].append(pred_class.item())
                answer3_gt_array[idx].append(gt_class.item())
                        
        for idx, label in enumerate(answer1_label_names):
            answer1_pred_target_label_array = answer1_pred_array_for_eval[idx]
            answer1_gt_target_label_array = answer1_gt_array[idx]
            label_acc_score = accuracy_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_precision_score = precision_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_recall_score = recall_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            label_f1_score = f1_score(answer1_gt_target_label_array, answer1_pred_target_label_array)
            total_f1_score += label_f1_score
            with open(cfg.Model_Save_Dir + "test_log.txt", "a") as f:
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", acc: " + str(label_acc_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", precision: " + str(label_precision_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", recall: " + str(label_recall_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", f1_score: " + str(label_f1_score) + "\n\n")

        answer2_acc_score = accuracy_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_precision_score = precision_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_recall_score = recall_score(answer2_gt_array, answer2_pred_array_for_eval)
        answer2_f1_score = f1_score(answer2_gt_array, answer2_pred_array_for_eval)
        total_f1_score += label_f1_score
        with open(cfg.Model_Save_Dir + "test_log.txt", "a") as f:
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", acc: " + str(answer2_acc_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", precision: " + str(answer2_precision_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", recall: " + str(answer2_recall_score) + "\n")
            f.write("epoch: " + str(epoch + 1) + ", 脑白质脱髓鞘" + ", f1_score: " + str(answer2_f1_score) + "\n\n")

        for idx, label in enumerate(answer3_label_names):
            answer3_pred_target_label_array = answer3_pred_array_for_eval[idx]
            answer3_gt_target_label_array = answer3_gt_array[idx]
            label_acc_score = accuracy_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_precision_score = precision_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_recall_score = recall_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            label_f1_score = f1_score(answer3_gt_target_label_array, answer3_pred_target_label_array)
            total_f1_score += label_f1_score
            with open(cfg.Model_Save_Dir + "test_log.txt", "a") as f:
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", acc: " + str(label_acc_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", precision: " + str(label_precision_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", recall: " + str(label_recall_score) + "\n")
                f.write("epoch: " + str(epoch + 1) + ", " + label + ", f1_score: " + str(label_f1_score) + "\n\n")
        
        total_f1_score /= 13
        
        if total_f1_score > best_f1_score:
            best_f1_score = total_f1_score
            torch.save(model.state_dict(), cfg.Model_Save_Dir + "save_model.pth")

def main():
    torch.manual_seed(717)
    torch.cuda.manual_seed(717)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    train_dataloader, test_dataloader = get_train_data_and_test_dataloader()

    model = MMCD()

    run(model, train_dataloader, test_dataloader)

if __name__ == "__main__":
    main()

