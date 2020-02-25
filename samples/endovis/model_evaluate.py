import numpy as np
import cv2
import glob

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_pre_rec_acc_F1(pre_mask,gt_mask):
    seg_inv, gt_inv = np.logical_not(pre_mask), np.logical_not(gt_mask)
    true_pos = float(np.logical_and(pre_mask, gt_mask).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pre_mask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt_mask).sum()

    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)

    return prec, rec, accuracy, F1

def compute_iou(pre_mask,gt_mask):
    """Calculates IoU of the given box with the array of the given boxes.
       pre_mask: [height, width]
       gt_mask: [height, width]
    """
    seg_inv, gt_inv = np.logical_not(pre_mask), np.logical_not(gt_mask)
    true_pos = float(np.logical_and(pre_mask, gt_mask).sum())  # float for division
    # true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(pre_mask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt_mask).sum()

    IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)

    return IoU

def comput_iou_per_instance(pred_masks, gt_masks, iou_threshold=0.5):
    """Calculates IoU of the given box with the array of the given boxes.
           pred_maskd: [height, width]
           gt_maskd: [height, width]
    """

    gt_match, pred_match, _ = compute_matches(pred_masks, gt_masks, iou_threshold)
    iou_dict = []
    for i in range(len(gt_match)):
        pre_mask = pred_masks[:, :, int(gt_match[i])]
        gt_mask = gt_masks[:, :, i]
        iou = compute_iou(pre_mask, gt_mask)

        iou_dict.append(iou)

    return iou_dict,len(iou_dict)


def compute_matches(pred_masks, gt_masks,iou_threshold=0.5, score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Pre def
    overlaps = np.ones([pred_masks.shape[-1],gt_masks.shape[-1]])
    pred_match = -1 * np.ones([pred_masks.shape[-1]])
    gt_match = -1 * np.ones([gt_masks.shape[-1]])

    for i in range(pred_masks.shape[-1]):
        for j in range(gt_masks.shape[-1]):

            # Compute IoU overlaps [pred_masks_i, gt_masks_j]
            IoU = compute_iou(pred_masks[:,:,i], gt_masks[:,:,j])
            overlaps[i,j] = IoU
    # print(overlaps)
    # Loop through predictions and find matching ground truth boxes
    match_count = 0

    for i in range(pred_masks.shape[-1]):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            match_count += 1
            gt_match[j] = i
            pred_match[i] = j
            break
    # # print(gt_match, pred_match, overlaps)
    # overlap_final = np.ones([pred_masks.shape[-1]])
    # for i in range(pred_masks.shape[-1]):
    #     overlap_final[i]= max(overlaps[i])

    return gt_match, pred_match, overlaps


def compute_ap( pred_masks,gt_masks,iou_threshold=0.5):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.(instance level)
    recalls: List of recall values at different class score thresholds.
    prec, rec : list of precisions and recalls at different class score thresholds.(pixel level)
    """
    # Get matches and overlaps
    gt_match, pred_match, _ = compute_matches(pred_masks, gt_masks,iou_threshold)

    ####### instace level

    # # Compute precision and recall at each prediction box step
    # n_precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    # n_recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)
    #
    # # Pad with start and end values to simplify the math
    # n_precisions = np.concatenate([[0], n_precisions, [0]])
    # n_recalls = np.concatenate([[0], n_recalls, [1]])
    #
    # # Ensure precision values decrease but don't increase. This way, the
    # # precision value at each recall threshold is the maximum it can be
    # # for all following recall thresholds, as specified by the VOC paper.
    # for i in range(len(n_precisions) - 2, -1, -1):
    #     n_precisions[i] = np.maximum(n_precisions[i], n_precisions[i + 1])
    #
    # # Compute mean AP over recall range
    # indices = np.where(n_recalls[:-1] != n_recalls[1:])[0] + 1
    # n_mAP = np.sum((n_recalls[indices] - n_recalls[indices - 1]) *
    #              n_precisions[indices])


    ####### pixel level
    prec, rec = [],[]

    for i in range(len(gt_match)):
        prec_i, rec_i, _,_ = compute_pre_rec_acc_F1(pred_masks[:,:,int(gt_match[i])], gt_masks[:,:,i])
        prec.append(prec_i)
        rec.append(rec_i)
    prec = np.concatenate([[0], prec, [0]])
    rec = np.concatenate([[0], rec, [1]])

    for i in range(len(prec) - 2, -1, -1):
        prec[i] = np.maximum(prec[i], prec[i + 1])

    indices = np.where(rec[:-1] != rec[1:])[0] + 1
    mAP = np.sum((rec[indices] - rec[indices - 1]) *
                 prec[indices])
    return mAP, prec, rec


def compute_ap_range(pred_masks,gt_masks,iou_thresholds=None, verbose=1):
    """Compute AP over a range or IoU thresholds. Default range is 0.5-0.95."""
    # Default is 0.5 to 0.95 with increments of 0.05
    iou_thresholds = iou_thresholds or np.arange(0.5, 1.0, 0.05)

    # Compute AP over range of IoU thresholds
    AP = []
    for iou_threshold in iou_thresholds:
        ap, precisions, recalls =compute_ap(pred_masks,gt_masks,iou_threshold=iou_threshold)
        # if verbose:
        #     print("AP @{:.2f}:\t {:.3f}".format(iou_threshold, ap))
        AP.append(ap)
    AP = np.array(AP).mean()
    # if verbose:
    #     print("AP @{:.2f}-{:.2f}:\t {:.3f}".format(
    #         iou_thresholds[0], iou_thresholds[-1], AP))
    return AP


def compute_Contours(pred_masks,gt_masks, iou_threshold=0.5):
    """Compute the contours difference.
        masks1, masks2: [Height, Width]
    """
    gt_match, pred_match, _ = compute_matches(pred_masks, gt_masks, iou_threshold)
    sim_dict = []
    for i in range(len(gt_match)):
        masks1 = pred_masks[:, :, int(gt_match[i])]
        masks2 = gt_masks[:, :, i]
        if masks1.sum()==0 or masks2.sum()==0:
            return 0,1
        a1 = cv2.copyMakeBorder(masks1, 0, 50, 0, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # gray = cv2.cvtColor(a1, cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(a1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours1 = contours[0]
        a2 = cv2.copyMakeBorder(masks2, 0, 50, 0, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # gray = cv2.cvtColor(a2, cv2.COLOR_BGR2GRAY)
        contours,_ = cv2.findContours(a2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2 = contours[0]

        sim_dict.append(cv2.matchShapes(contours1, contours2, 1, 0))

    return sim_dict, len(sim_dict)

def compute_dice_coefficient(pred_masks,gt_masks, iou_threshold=0.5):
    """Compute soerensen-dice coefficient.

    compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
    and the predicted mask `mask_pred`.

    Args:
      mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
      mask_pred: 3-dim Numpy array of type bool. The predicted mask.

    Returns:
      dice_dict is a list of each instance dice value.
      the dice coeffcient as float. If both masks are empty, the result is NaN.
    """
    gt_match, pred_match, _ = compute_matches(pred_masks, gt_masks, iou_threshold)
    dice_dict = []
    for i in range(len(gt_match)):
        mask_pred = pred_masks[:,:,int(gt_match[i])]
        mask_gt = gt_masks[:,:,i]

        volume_sum = mask_gt.sum() + mask_pred.sum()
        if mask_gt.sum() == 0:
            return 0,1
        volume_intersect = (mask_gt & mask_pred).sum()
        dice_dict.append(2 * volume_intersect / volume_sum)

    return dice_dict, len(dice_dict)

def testEachimage(pre_list,gt_list):
    pre_mask = np.zeros((1024, 1280, len(pre_list))).astype(np.uint8)
    gt_mask = np.zeros((1024, 1280, len(gt_list))).astype(np.uint8)
    for i in range(len(pre_list)):
        mask_i = cv2.cvtColor(cv2.imread(pre_list[i]), cv2.COLOR_RGB2GRAY)
        pre_mask[:, :, i] = mask_i
    for i in range(len(gt_list)):
        mask_i = cv2.cvtColor(cv2.imread(gt_list[i]), cv2.COLOR_RGB2GRAY)
        gt_mask[:, :, i] = mask_i

    mAP, precisions, recalls = compute_ap(pre_mask, gt_mask)

    AP = compute_ap_range(pre_mask, gt_mask)

    dice,num_dice = compute_dice_coefficient(pre_mask, gt_mask)
    m_dice = np.array(dice).sum()/num_dice

    contour_sim,num_sim = compute_Contours(pre_mask, gt_mask)
    m_sim = np.array(contour_sim).sum()/num_sim

    iou,num_iou = comput_iou_per_instance(pre_mask, gt_mask)
    m_iou = np.array(iou).sum()/num_iou

    return mAP, AP, dice, m_dice, contour_sim, m_sim, iou, m_iou

def testEachimage_for_test(pre_masks,gt_list):
    pre_mask = np.zeros((1024, 1280, len(pre_masks))).astype(np.uint8)
    gt_mask = np.zeros((1024, 1280, len(gt_list))).astype(np.uint8)
    for i in range(len(pre_masks)):
        mask_i = pre_masks[i]
        pre_mask[:, :, i] = mask_i
    for i in range(len(gt_list)):
        mask_i = cv2.cvtColor(cv2.imread(gt_list[i]), cv2.COLOR_RGB2GRAY)
        gt_mask[:, :, i] = mask_i

    mAP, precisions, recalls = compute_ap(pre_mask, gt_mask)

    AP = compute_ap_range(pre_mask, gt_mask)

    return mAP, AP



if __name__ == '__main__':

    # pre = ['/Users/shiwakaga/Amodel_instrument/output_mrcnn/instrument10/frame001_ins_0.png']
    # gt = ['/Users/shiwakaga/Amodel_Data/test/instrument10/amodel/frame001_ins_0.png']
    #
    # mAP, AP, dice, m_dice, contour_sim, m_sim, iou, m_iou = testEachimage(pre, gt)
    # print(mAP, AP, dice, m_dice, contour_sim, m_sim, iou, m_iou)

    train_list = glob.glob('/Users/shiwakaga/Amodel_Data/train/*/images/*.jpg')
    test_list = glob.glob('/Users/shiwakaga/Amodel_Data/test/*/images/*.png')
    print('All evaluate train image num is : ',len(train_list))
    print('All evaluate test image num is : ', len(test_list))

    all_result = {'mrcnn':[],'mrcnn_70':[],'1':[],'2':[],'3':[]}
    modes = ['test','train']

    for mode  in modes:
        for key in all_result.keys():
            pre_path ='/Users/shiwakaga/Amodel_instrument/output_{}/'.format(key)

            with open('result_{}_{}.txt'.format(mode,key), 'a') as f:
                # f.writelines('video_id img_id mAP AP dice m_dice contour_sim m_sim' + '\n')

                result = {'mAP': [], 'mdice': [], 'msim': [], 'mIOU': []}
                if mode == 'train':
                    list = train_list
                else:
                    list = test_list
                for img in list:
                    print(img)
                    img_id = img.split('/')[-1][:-4]
                    img_ins_id = img.split('/')[-3]

                    pre_list = glob.glob(pre_path + img_ins_id + '/' + img_id +'*.png')
                    gt_list = glob.glob('/Users/shiwakaga/Amodel_Data/'+str(mode)+'/'+img_ins_id+'/amodel/'+img_id +'*.png')


                    if len(pre_list)==0 and len(gt_list)==0:
                        f.writelines(img_ins_id + ' ' + img_id + '1 1 1 1 0 0 1 1 \n')
                        result['mAP'].append(1)
                        result['mdice'].append(1)
                        result['msim'].append(0)
                        result['mIOU'].append(1)
                    else:
                        if len(pre_list)==0 and len(gt_list)!=0:
                            f.writelines(img_ins_id + ' ' + img_id + '0 0 0 0 1 1 0 0 \n')
                            result['mAP'].append(0)
                            result['mdice'].append(0)
                            result['msim'].append(1)
                            result['mIOU'].append(0)
                        else:
                            if len(pre_list)!=0 and len(gt_list)==0:
                                f.writelines(img_ins_id + ' ' + img_id + '0 0 0 0 1 1 0 0 \n')
                                result['mAP'].append(0)
                                result['mdice'].append(0)
                                result['msim'].append(1)
                                result['mIOU'].append(0)
                            else:
                                mAP, AP, dice, m_dice, contour_sim, m_sim, iou, m_iou = testEachimage(pre_list, gt_list)
                                result['mAP'].append(mAP)
                                result['mdice'].append(m_dice)
                                result['msim'].append(m_sim)
                                result['mIOU'].append(m_iou)

                                f.writelines(img_ins_id +' '+ img_id +' '+ str(mAP) +' '+ str(AP) +' '
                                             + str(dice) +' '+ str(m_dice) +' '+ str(contour_sim) +' '
                                             + str(m_sim) +' '+ str(iou) +' '+ str(m_iou)  +'\n')
                for t in result.keys():
                    result[t] = np.array(result[t]).sum()/len(train_list)
                all_result[key].append(result)
    with open('all_result.txt''a') as f:
        for key, value in all_result.items():
            print(key + ": " + str(value))
            f.writelines(key + ": " + str(value)+'\n')


