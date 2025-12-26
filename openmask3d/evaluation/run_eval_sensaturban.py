import os
import numpy as np
import clip
import torch
from eval_sensatUrban import evaluate
from sensatUrban_constants import VALID_CLASS_IDS, CLASS_LABELS, COLOR_MAP
import tqdm
import argparse
from glob import glob

class SensatUrbanInstSegEvaluator():
    def __init__(self, clip_model_type, sentence_structure="a {} in a scene"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_type = clip_model_type
        self.clip_model = self.get_clip_model(clip_model_type)
        self.query_sentences = self.get_query_sentences(sentence_structure)
        self.feature_size = self.get_feature_size(clip_model_type)
        self.text_query_embeddings = self.get_text_query_embeddings().numpy()
        self.set_label_and_color_mapper()

    def get_query_sentences(self, sentence_structure="a {} in a scene"):
        label_list = list(CLASS_LABELS)
        return [sentence_structure.format(label) for label in label_list]

    def get_clip_model(self, clip_model_type):
        clip_model, _ = clip.load(clip_model_type, self.device)
        return clip_model

    def get_feature_size(self, clip_model_type):
        if clip_model_type == 'ViT-L/14' or clip_model_type == 'ViT-L/14@336px':
            return 768
        elif clip_model_type == 'ViT-B/32':
            return 512
        else:
            raise NotImplementedError(f"Unsupported CLIP model type: {clip_model_type}")

    def get_text_query_embeddings(self):
        text_query_embeddings = torch.zeros((len(self.query_sentences), self.feature_size))

        for label_idx, sentence in enumerate(self.query_sentences):
            text_input_processed = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                sentence_embedding = self.clip_model.encode_text(text_input_processed)

            sentence_embedding_normalized = (sentence_embedding/sentence_embedding.norm(dim=-1, keepdim=True)).float().cpu()
            text_query_embeddings[label_idx, :] = sentence_embedding_normalized

        return text_query_embeddings
    
    def set_label_and_color_mapper(self):
        # 创建标签映射：索引 -> 类别ID
        self.label_mapper = np.vectorize({idx: el for idx, el in enumerate(VALID_CLASS_IDS)}.get)
        self.color_mapper = np.vectorize(COLOR_MAP.get)

    def compute_classes_per_mask(self, masks_path, mask_features_path, keep_first=None):
        masks = torch.load(masks_path)
        mask_features = np.load(mask_features_path)

        if keep_first is not None:
            masks = masks[:, 0:keep_first]
            mask_features = mask_features[0:keep_first, :]

        # normalize mask features
        mask_features_normalized = mask_features/np.linalg.norm(mask_features, axis=1)[..., None]

        similarity_scores = mask_features_normalized@self.text_query_embeddings.T
        max_class_similarity_scores = np.max(similarity_scores, axis=1)
        max_ind = np.argmax(similarity_scores, axis=1)
        max_ind_remapped = self.label_mapper(max_ind)
        pred_classes = max_ind_remapped

        return masks, pred_classes, max_class_similarity_scores

    def evaluate_scenes(self, mask_pred_dir, mask_features_dir, gt_dir):
        """
        评估所有场景的预测结果
        """
        # 获取所有掩码文件
        mask_files = sorted(glob(os.path.join(mask_pred_dir, '*_masks.pt')))
        
        preds = {}
        for mask_file in tqdm.tqdm(mask_files, desc="Computing predictions"):
            scene_name = os.path.basename(mask_file).replace('_masks.pt', '')
            
            # 查找对应的特征文件
            feature_file = os.path.join(mask_features_dir, f"{scene_name}_openmask3d_features.npy")
            if not os.path.exists(feature_file):
                print(f"[WARNING] Feature file not found for {scene_name}, skipping...")
                continue
            
            masks, pred_classes, pred_scores = self.compute_classes_per_mask(mask_file, feature_file)
            
            # 转换为评估格式
            pred_masks_bool = masks.numpy().astype(bool)  # (num_points, num_masks)
            pred_info = {
                'pred_classes': pred_classes,
                'pred_scores': pred_scores,
                'pred_masks': pred_masks_bool
            }
            preds[scene_name] = pred_info
        
        # 运行评估
        print(f"[INFO] Evaluating {len(preds)} scenes...")
        results = evaluate(preds, gt_dir)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenMask3D on SensatUrban dataset')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth directory')
    parser.add_argument('--mask_pred_dir', type=str, required=True, help='Path to predicted masks directory')
    parser.add_argument('--mask_features_dir', type=str, required=True, help='Path to mask features directory')
    parser.add_argument('--clip_model', type=str, default='ViT-L/14@336px', help='CLIP model type')
    parser.add_argument('--sentence_structure', type=str, default='a {} in a scene', help='Sentence structure for CLIP queries')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = SensatUrbanInstSegEvaluator(
        clip_model_type=args.clip_model,
        sentence_structure=args.sentence_structure
    )
    
    # 运行评估
    results = evaluator.evaluate_scenes(
        mask_pred_dir=args.mask_pred_dir,
        mask_features_dir=args.mask_features_dir,
        gt_dir=args.gt_dir
    )
    
    print("[INFO] Evaluation completed!")


if __name__ == "__main__":
    main()

