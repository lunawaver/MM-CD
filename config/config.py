ConvNeXT_NAME = "facebook/convnextv2-tiny-22k-224"

Bert_NAME = "google-bert/bert-base-chinese"

Train_Image_Root_Folder = "./data/train/train_images"
Test_Image_Root_Folder = "./data/test/test_images"

Train_Report_CSV_File = "./data/train/train_reports/train_reports.csv"
Test_Report_CSV_File = "./data/test/test_reports/test_reports.csv"

Train_Answer1_CSV_File = "./data/train/train_answers/train_Q1_Answer.csv"
Train_Answer2_CSV_File = "./data/train/train_answers/train_Q2_Answer.csv"
Train_Answer3_CSV_File = "./data/train/train_answers/train_Q3_Answer.csv"
Test_Answer1_CSV_File = "./data/test/test_answers/test_Q1_Answer.csv"
Test_Answer2_CSV_File = "./data/test/test_answers/test_Q2_Answer.csv"
Test_Answer3_CSV_File = "./data/test/test_answers/test_Q3_Answer.csv"

Answer1_Label_Num = 7
Answer2_Class_Num = 2
Answer3_Label_Num = 5

device = "cuda:0"

epoch_num = 100

Patient_Batch_Size = 1

Bert_Max_Length = 512

Report_Max_Length = 300
Question_Max_Length = 21
Answer_Max_Length = 4
Pad_Length = 512 - ((Report_Max_Length + 1) + 3 * (Question_Max_Length + 1) + 3 * (Answer_Max_Length + 1))

Mask1_Idx = (Report_Max_Length + 1) + (Question_Max_Length + 1) + Answer_Max_Length
Mask2_Idx = (Report_Max_Length + 1) + (Question_Max_Length + 1) * 2 + (Answer_Max_Length + 1) + Answer_Max_Length
Mask3_Idx = (Report_Max_Length + 1) + (Question_Max_Length + 1) * 3 + (Answer_Max_Length + 1) * 2 + Answer_Max_Length

Model_Save_Dir = "./model_save_dir/"
