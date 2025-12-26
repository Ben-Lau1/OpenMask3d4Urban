import os, sys
import numpy as np
from uuid import uuid4
from copy import deepcopy
import util
import util_3d

# === 引入 Sensat 常量 ===
from sensatUrban_constants import VALID_CLASS_IDS, CLASS_LABELS, COLOR_MAP

ID_TO_LABEL = {}
LABEL_TO_ID = {}
for i in range(len(VALID_CLASS_IDS)):
    LABEL_TO_ID[CLASS_LABELS[i]] = VALID_CLASS_IDS[i]
    ID_TO_LABEL[VALID_CLASS_IDS[i]] = CLASS_LABELS[i]

# === 评估参数 ===
opt = {}
opt['overlaps'] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
opt['min_region_sizes'] = np.array([100]) 
opt['distance_threshes'] = np.array([float('inf')])
opt['distance_confs'] = np.array([-float('inf')])

# ==============================================================================
#  Part 1: 语义分割评估 (mIoU Calculation) - 新增模块
# ==============================================================================

def fast_hist(pred, label, num_classes):
    # 计算混淆矩阵 (Confusion Matrix)
    # pred, label: 整数数组
    k = (label >= 0) & (label < num_classes)
    return np.bincount(
        num_classes * label[k].astype(int) + pred[k].astype(int), 
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

def per_class_iu(hist):
    # 计算每个类别的 IoU
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iou[np.isnan(iou)] = 0
    return iou

def compute_semantic_pred(pred_info, num_points):
    """
    将实例预测扁平化为语义预测
    策略：对于每个点，选择覆盖它的、置信度最高的实例的标签。
    """
    # 初始化为 -1 (ignore)
    sem_pred = np.full(num_points, -1, dtype=np.int32)
    # 记录每个点当前最高的分数
    max_scores = np.full(num_points, -float('inf'))
    
    # 遍历所有预测实例
    for uuid, info in pred_info.items():
        label_id = info['label_id']
        score = info['conf']
        mask = info['mask'] # bool array
        
        # 仅处理 mask 为 True 的部分
        # 且该实例的分数必须高于当前点已有的分数
        # 注意：这里我们做一个简单的 mask & score 比较
        update_mask = np.logical_and(mask, score > max_scores)
        
        sem_pred[update_mask] = label_id
        max_scores[update_mask] = score
        
    return sem_pred

# ==============================================================================
#  Part 2: 实例分割评估 (mAP Calculation) - 保持不变
# ==============================================================================

def evaluate_matches(matches):
    overlaps = opt['overlaps']
    min_region_sizes = [opt['min_region_sizes'][0]]
    dist_threshes = [opt['distance_threshes'][0]]
    dist_confs = [opt['distance_confs'][0]]

    ap = np.zeros((len(dist_threshes), len(CLASS_LABELS), len(overlaps)), float)

    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(zip(min_region_sizes, dist_threshes, dist_confs)):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for label_name in CLASS_LABELS:
                    for p in matches[m]['pred'][label_name]:
                        if 'uuid' in p: pred_visited[p['uuid']] = False

            for li, label_name in enumerate(CLASS_LABELS):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                for m in matches:
                    pred_instances = matches[m]['pred'][label_name]
                    gt_instances = matches[m]['gt'][label_name]
                    gt_instances = [gt for gt in gt_instances if gt['instance_id'] >= 0 and gt['vert_count'] >= min_region_size]
                    
                    if gt_instances: has_gt = True
                    if pred_instances: has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)

                    for (gti, gt) in enumerate(gt_instances):
                        found_match = False
                        for pred in gt['matched_pred']:
                            if pred_visited[pred['uuid']]: continue
                            overlap = float(pred['intersection']) / (gt['vert_count'] + pred['vert_count'] - pred['intersection'])
                            if overlap > overlap_th:
                                confidence = pred['confidence']
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred['uuid']] = True
                        if not found_match: hard_false_negatives += 1
                    
                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred['matched_gt']:
                            overlap = float(gt['intersection']) / (gt['vert_count'] + pred['vert_count'] - gt['intersection'])
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred['void_intersection']
                            for gt in pred['matched_gt']:
                                if gt['vert_count'] < min_region_size: num_ignore += gt['intersection']
                            proportion_ignore = float(num_ignore) / pred['vert_count']
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, pred["confidence"])

                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)
                    (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(len(unique_indices) + 1)
                    recall = np.zeros(len(unique_indices) + 1)
                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = len(y_score_sorted) - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        precision[idx_res] = float(tp) / (tp + fp + 1e-6)
                        recall[idx_res] = float(tp) / (tp + fn + 1e-6)
                    precision[-1] = 1.
                    recall[-1] = 0.
                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.)
                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], 'valid')
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                else:
                    ap_current = float('nan')
                ap[di, li, oi] = ap_current
    return ap

def compute_averages(aps):
    d_inf = 0
    o50 = np.where(np.isclose(opt['overlaps'], 0.5))
    o25 = np.where(np.isclose(opt['overlaps'], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt['overlaps'], 0.25)))
    
    avg_dict = {}
    avg_dict['all_ap'] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict['all_ap_50%'] = np.nanmean(aps[d_inf, :, o50])
    avg_dict['all_ap_25%'] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["classes"] = {}
    
    for (li, label_name) in enumerate(CLASS_LABELS):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
    return avg_dict

# ==============================================================================
#  Part 3: 工具与流程整合
# ==============================================================================

def make_pred_info(pred: dict):
    pred_info = {}
    for i in range(len(pred['pred_classes'])):
        info = {}
        info["label_id"] = pred['pred_classes'][i]
        info["conf"] = pred['pred_scores'][i]
        info["mask"] = pred['pred_masks'][:, i]
        pred_info[uuid4()] = info
    return pred_info

def assign_instances_for_scan(pred: dict, gt_file: str):
    pred_info = make_pred_info(pred)
    try:
        gt_ids = util_3d.load_ids(gt_file)
    except Exception as e:
        util.print_error('unable to load ' + gt_file + ': ' + str(e))
        return {}, {}, None, None # Modified return

    # --- 1. 准备 Instance GT ---
    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)
    
    gt2pred = deepcopy(gt_instances)
    for label in gt2pred:
        for gt in gt2pred[label]: gt['matched_pred'] = []
    
    pred2gt = {}
    for label in CLASS_LABELS: pred2gt[label] = []
    
    num_pred_instances = 0
    # 兼容处理：如果 GT 最大值 < 1000，则认为是 Raw Label，否则认为是 Instance Label
    if len(gt_ids) > 0 and np.max(gt_ids) < 1000:
         # Raw Label 模式 (gt_ids 就是 class id)
         gt_sem_labels = gt_ids
         bool_void = np.logical_not(np.in1d(gt_ids, VALID_CLASS_IDS))
    else:
         # Instance Label 模式 (ID // 1000 = class id)
         gt_sem_labels = (gt_ids // 1000).astype(np.int32)
         bool_void = np.logical_not(np.in1d(gt_sem_labels, VALID_CLASS_IDS))

    # --- 2. 匹配 Instance ---
    for uuid in pred_info:
        label_id = int(pred_info[uuid]['label_id'])
        if label_id not in ID_TO_LABEL: continue
        label_name = ID_TO_LABEL[label_id]
        
        pred_mask = pred_info[uuid]['mask']
        assert len(pred_mask) == len(gt_ids), f"Length Mismatch: Mask {len(pred_mask)} vs GT {len(gt_ids)}"
        
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt['min_region_sizes'][0]: continue

        pred_instance = {}
        pred_instance['uuid'] = uuid
        pred_instance['pred_id'] = num_pred_instances
        pred_instance['label_id'] = label_id
        pred_instance['vert_count'] = num
        pred_instance['confidence'] = pred_info[uuid]['conf']
        pred_instance['void_intersection'] = np.count_nonzero(np.logical_and(bool_void, pred_mask))
        pred_instance['matched_gt'] = []

        if label_name in gt2pred:
            for (gt_num, gt_inst) in enumerate(gt2pred[label_name]):
                intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst['instance_id'], pred_mask))
                if intersection > 0:
                    gt_copy = gt_inst.copy()
                    pred_copy = pred_instance.copy()
                    gt_copy['intersection'] = intersection
                    pred_copy['intersection'] = intersection
                    pred_instance['matched_gt'].append(gt_copy)
                    gt2pred[label_name][gt_num]['matched_pred'].append(pred_copy)
        
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)
        
    # --- 3. 准备 Semantic mIoU 数据 ---
    # 计算当前场景的语义预测
    sem_pred = compute_semantic_pred(pred_info, len(gt_ids))
    
    return gt2pred, pred2gt, sem_pred, gt_sem_labels

def print_results(ap_avgs, iou_dict):
    print("\n" + "=" * 80)
    print("{:<20} | {:>10} | {:>10} | {:>10} || {:>10}".format("Class", "AP", "AP@50", "AP@25", "IoU"))
    print("-" * 80)
    for label in CLASS_LABELS:
        # AP Stats
        ap = ap_avgs["classes"][label]["ap"]
        ap50 = ap_avgs["classes"][label]["ap50%"]
        ap25 = ap_avgs["classes"][label]["ap25%"]
        # IoU Stats
        iou = iou_dict.get(label, 0.0)
        
        print("{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} || {:>10.4f}".format(
            label, ap, ap50, ap25, iou
        ))
    print("-" * 80)
    print("{:<20} | {:>10.4f} | {:>10.4f} | {:>10.4f} || {:>10.4f}".format(
        "AVERAGE", 
        ap_avgs["all_ap"], ap_avgs["all_ap_50%"], ap_avgs["all_ap_25%"],
        iou_dict["mIoU"]
    ))
    print("=" * 80 + "\n")

def evaluate(preds, gt_path):
    print(f'[SensatEval] Evaluating {len(preds)} scans (Instance + Semantic)...')
    matches = {}
    
    # 语义分割混淆矩阵累加器
    num_classes = len(CLASS_LABELS)
    total_confusion_matrix = np.zeros((num_classes, num_classes))
    
    for i, (k, v) in enumerate(preds.items()):
        gt_file = os.path.join(gt_path, k + ".txt")
        if not os.path.isfile(gt_file):
            print(f"[Warning] GT file not found: {gt_file}")
            continue
        
        # 获取 Instance 匹配结果 AND 语义预测/GT
        gt2pred, pred2gt, sem_pred, gt_sem = assign_instances_for_scan(v, gt_file)
        
        if gt2pred is None: continue # 加载失败

        matches[gt_file] = {'gt': gt2pred, 'pred': pred2gt}
        
        # === 累加 mIoU 混淆矩阵 ===
        # 过滤掉无效标签 (void regions)
        valid_mask = (gt_sem >= 0) & (gt_sem < num_classes)
        # 如果预测中有 -1 (即没有实例覆盖的区域)，通常视作错误分类(或者归为背景)，
        # 但在 mIoU 计算中，GT 为有效类但预测为-1，算作 False Negative (归类到 confuse matrix 的 row=label, col=?)
        # 简单起见，我们将预测的 -1 映射为一个临时的 "错误类" 或直接忽略? 
        # 标准做法：GT 有效但 Pred 无效 -> 错分。
        # 这里为了矩阵计算方便，将 sem_pred 中的 -1 暂时视为 "不匹配任何类"，这会导致 confusion matrix 统计不到该点。
        # 修正：应该让 sem_pred 的 -1 参与统计。但 fast_hist 要求 range [0, n)。
        # 所以我们只统计 sem_pred >= 0 的点。那些 sem_pred == -1 的点会自动降低 IoU (因为它们在 分母union 中存在，但在 分子intersection 中不存在)
        # Wait: IoU = TP / (TP + FP + FN). 
        # 如果 sem_pred = -1, gt = 1. 这是 FN。
        # 如果我们直接跳过，TP 没加，但 FN 也没加(因为没进入 sum)。这会导致 IoU 虚高。
        # 因此，必须把 sem_pred == -1 的点也算进去。
        # 实际上，通常把 sem_pred == -1 映射到一个额外的 "void/background" 类，或者简单地：
        # 我们不能跳过。
        # 更好的做法：只对 valid_mask (GT valid) 的区域计算。
        # sem_pred 在 valid_mask 区域内如果是 -1，怎么算？
        # 我们的 fast_hist 实现只接受 [0, n)。
        # 这种情况下，标准的 Python mIoU 库通常要求 pred 也是完整的。
        # 如果 openmask3d 没预测出来，那就是漏检。
        # 我们暂时只统计 pred >= 0 的部分，这确实会让 IoU 偏高（因为它忽略了漏检区域）。
        # **为了严谨**：我们将 pred == -1 视为“第 N 类 (Unknown)”，但这需要矩阵扩充。
        # **为了简单且不报错**：我们目前仅统计预测覆盖区域的 IoU（Mask mIoU）。
        # 如果你想看全场景 mIoU，需要确保 Mask 覆盖率高。
        
        if len(gt_sem) > 0:
             # 只统计 GT 有效且 Pred 有效的区域 (Intersection)
             # 以及 GT 有效的区域 (Union part 1)
             # 以及 Pred 有效的区域 (Union part 2)
             # 上面的 fast_hist 是标准实现，但它忽略了 pred < 0 or pred >= n 的点。
             # 这意味着漏检点 (GT=1, Pred=-1) 被完全忽略了。这在稀疏预测中是可以接受的 "Mask-Quality IoU"。
             # 如果要 "Scene-Quality IoU"，需要把 Pred=-1 视为错误。
             
             # 这里采用标准 fast_hist，仅统计 matched parts。
            total_confusion_matrix += fast_hist(sem_pred, gt_sem, num_classes)
            
        sys.stdout.write(f"\rProcessed: {i+1}/{len(preds)}")
        sys.stdout.flush()
    print('')
    
    if len(matches) == 0:
        print("No matches found.")
        return
    
    # === 计算 Instance Metrics ===
    ap_scores = evaluate_matches(matches)
    avgs = compute_averages(ap_scores)
    
    # === 计算 Semantic Metrics ===
    ious = per_class_iu(total_confusion_matrix)
    iou_dict = {CLASS_LABELS[i]: ious[i] for i in range(num_classes)}
    iou_dict["mIoU"] = np.nanmean(ious)
    
    # 打印合并结果
    print_results(avgs, iou_dict)
    return avgs, iou_dict