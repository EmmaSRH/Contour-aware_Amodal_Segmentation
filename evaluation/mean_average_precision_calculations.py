import numpy as np
from pandas import DataFrame
from scipy.optimize import linear_sum_assignment as hungarian_algorithm


def compute_iou(mask_gt, mask_pred):
    """
        Compute the intersection over union (https://en.wikipedia.org/wiki/Jaccard_index)

        compute the intersectin over union between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred`.

        Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

        Returns:
        the iou coeffcient as float. If both masks are empty, the result is 0
    """
    mask_gt = mask_gt.astype('bool')
    mask_pred = mask_pred.astype('bool')
    overlap = mask_gt * mask_pred  # Logical AND
    union = mask_gt + mask_pred  # Logical OR
    iou = overlap.sum() / float(union.sum())  # Treats "True" as 1,
    return iou


def compute_statistics(mask_gt, mask_pred):
    """
        Compute Statistic

        compute statistics (TP, FP, FN, precision, recall) between the ground truth mask `mask_gt`
        and the predicted mask `mask_pred`.
        TP = True positive (defined as an iou>=0.03)
        FP = False positive
        FN = False negative
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        Args:
        mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
        mask_pred: 3-dim Numpy array of type bool. The predicted mask.

        Returns:
        output = dict(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
            precision=precision,
            recall=recall
        )
    """
    # define constants
    min_iou_for_match = 0.03

    # get number of labels in image
    instances_gt = list(np.unique(mask_gt))
    instances_pred = list(np.unique(mask_pred))

    # remove background
    instances_gt = instances_gt[1:]
    instances_pred = instances_pred[1:]

    # create performance matrix
    performance_matrix = np.zeros((len(instances_gt), len(instances_pred)))
    masks = []

    # calculate dice score for each ground truth to predicted instance
    for counter_gt, instance_gt in enumerate(instances_gt):

        # create binary mask for current gt instance
        gt = mask_gt.copy()
        gt[mask_gt != instance_gt] = 0
        gt[mask_gt == instance_gt] = 1

        masks_row = []
        for counter_pred, instance_pred in enumerate(instances_pred):
            # make binary mask for current predicted instance
            prediction = mask_pred.copy()
            prediction[mask_pred != instance_pred] = 0
            prediction[mask_pred == instance_pred] = 1

            # calculate iou
            # show_image(gt, prediction)
            iou = compute_iou(gt, prediction)
            performance_matrix[counter_gt, counter_pred] = iou
            masks_row.append([gt, prediction])
        masks.append(masks_row)

    # delete all matches smaller than threshold
    performance_matrix[performance_matrix < min_iou_for_match] = 0

    # assign instrument instances according to hungarian algorithm
    label_assignment = hungarian_algorithm(performance_matrix * -1)
    label_nr_gt, label_nr_pred = label_assignment

    # get performance per instance

    true_positive_list = []
    for i in range(len(label_nr_gt)):
        instance_iou = performance_matrix[label_nr_gt[i], label_nr_pred[i]]
        true_positive_list.append(instance_iou)
    true_positive_list = list(filter(lambda a: a != 0, true_positive_list))  # delete all 0s assigned to a label

    true_positive = len(true_positive_list)
    false_negative = len(instances_gt) - true_positive
    false_positive = len(instances_pred) - true_positive

    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = 0
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = 0

    output = dict(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall
    )

    return output


def compute_mean_average_precision(statistic_list):
    """
        Compute the mean average precision:
        (https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)

        We define average precision as Area under Curve AUC)
        https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

        Args:
        statistic_list: 1-dim list, containing statistics dicts (dict definition, see function compute_statistics).

        Returns:
        the area_under_curve as float
        )
    """

    # create data frame
    data_frame = DataFrame(columns=["true_positive", "false_positive", "false_negative", "precision", "recall"])

    # add data
    data_frame = data_frame.append(statistic_list)
    data_frame = data_frame.reset_index()

    # interpolate precision with highest recall for precision
    data_frame = data_frame.sort_values(by="recall", ascending=False)
    precision_interpolated = []
    current_highest_value = 0
    for index, row in data_frame.iterrows():
        if row.precision > current_highest_value:
            current_highest_value = row.precision
        precision_interpolated.append(current_highest_value)
    data_frame['precision_interpolated'] = precision_interpolated

    # get changes in interpolated precision curve
    data_frame_grouped = data_frame.groupby("recall")
    changes = []
    for item in data_frame_grouped.groups.items():
        current_recall = item[0]
        idx_precision = item[1][0]
        current_precision_interpolated = data_frame.loc[idx_precision].precision_interpolated
        change = dict(recall=current_recall, precision_interpolated=current_precision_interpolated)
        changes.append(change)
    # add end and starting point
    if changes[0]["recall"] != 0.0:
        changes.insert(0, dict(recall=0, precision_interpolated=changes[0]["precision_interpolated"]))
    if current_recall < 1:
        changes.append(dict(recall=1, precision_interpolated=current_precision_interpolated))

    # calculate area under curve
    area_under_curve = 0
    for i in range(1, len(changes)):
        precision_area = (changes[i]["recall"] - changes[i - 1]["recall"]) * changes[i]["precision_interpolated"]
        area_under_curve += precision_area

    return area_under_curve
