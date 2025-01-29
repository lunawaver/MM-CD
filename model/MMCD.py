import torch
import torch.nn as nn

import config.config as cfg

from model.ConvNext import ImageEncoder
from model.MultiImage_Weight_Generator import MultiImageWeightGenerator
from model.Bert import ReportEncoder

from model.MultiImage_Fusion import MultiImageFusionModel
from model.MultiModal_Fusion import MultiModelFusionModel

class MMCD(nn.Module):
    def __init__(self):
        super(MMCD, self).__init__()
        self.image_encoder = ImageEncoder().to("cuda")

        self.multi_image_weight_generator_model = MultiImageWeightGenerator().to("cuda")

        self.multi_image_fusion_model = MultiImageFusionModel().to("cuda")

        self.report_encoder = ReportEncoder().to("cuda")

        self.multi_model_fusion_model = MultiModelFusionModel(768, 12).to("cuda")

        self.pred_answer1_none_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_lacunar_infarction_lesion_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_ischemic_lesion_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_softening_lesion_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_lacunar_lesion_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_infarction_lesion_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer1_hemorrhagic_lesion_layer = nn.Linear(768, 2).to("cuda")

        self.pred_answer2_layer = nn.Linear(768, 2).to("cuda")

        self.pred_answer3_none_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer3_fptoi_lobe_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer3_basal_ganglia_and_corona_radiata_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer3_cerebral_hemisphere_and_cerebellar_hemisphere_layer = nn.Linear(768, 2).to("cuda")
        self.pred_answer3_other_layer = nn.Linear(768, 2).to("cuda")

    def forward(self, images, input_ids, token_type_ids, attention_mask):
        pred_answer1_list = []
        pred_answer2_list = []
        pred_answer3_list = []
        image_encoder_output = self.image_encoder(images).flatten(2)

        weight_generator_input = image_encoder_output.permute(0, 2, 1)

        predict_matrix = self.multi_image_weight_generator_model(weight_generator_input)

        image_encoder_output = image_encoder_output * predict_matrix.view(image_encoder_output.shape[0], 1, 1)

        multi_image_fusion_output = self.multi_image_fusion_model(image_encoder_output)

        report_encoder_output = self.report_encoder(input_ids, token_type_ids, attention_mask)

        multi_model_concat_output = torch.cat((report_encoder_output, multi_image_fusion_output), dim=1)

        multi_model_fusion_output = self.multi_model_fusion_model(multi_model_concat_output).permute(1, 0, 2).squeeze(0)

        answer1_mask_vector = multi_model_fusion_output[cfg.Mask1_Idx]
        answer2_mask_vector = multi_model_fusion_output[cfg.Mask2_Idx]
        answer3_mask_vector = multi_model_fusion_output[cfg.Mask3_Idx]

        pred_answer1_list.append(self.pred_answer1_none_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_lacunar_infarction_lesion_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_ischemic_lesion_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_softening_lesion_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_lacunar_lesion_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_infarction_lesion_layer(answer1_mask_vector))
        pred_answer1_list.append(self.pred_answer1_hemorrhagic_lesion_layer(answer1_mask_vector))

        pred_answer2 = self.pred_answer2_layer(answer2_mask_vector)

        pred_answer3_list.append(self.pred_answer3_none_layer(answer3_mask_vector))
        pred_answer3_list.append(self.pred_answer3_fptoi_lobe_layer(answer3_mask_vector))
        pred_answer3_list.append(self.pred_answer3_basal_ganglia_and_corona_radiata_layer(answer3_mask_vector))
        pred_answer3_list.append(self.pred_answer3_cerebral_hemisphere_and_cerebellar_hemisphere_layer(answer3_mask_vector))
        pred_answer3_list.append(self.pred_answer3_other_layer(answer3_mask_vector))

        return pred_answer1_list, pred_answer2, pred_answer3_list
