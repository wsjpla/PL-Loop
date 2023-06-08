import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch

from superglue.matching import Matching
from superglue.utils import (make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)

def loop_verification(img_fname_1, img_fname_2, new_size=None, is_vis=False, crop=True):
    # 读取两幅图像
    image1 = cv2.imread(img_fname_1, 0)
    if new_size is not None:
        image1 = cv2.resize(image1, new_size)
    image2 = cv2.imread(img_fname_2, 0)
    if new_size is not None:
        image2 = cv2.resize(image2, new_size)

    # crop: 对Robotar数据集中的图像进行裁剪，剪掉下半部分车的部分
    if crop == True:
        img1 = image1[0:400, :]
        img2 = image2[0:400, :]
    else:
        img1 = image1
        img2 = image2
        
    # superpoint+superglue
    keys = ['keypoints', 'scores', 'descriptors']

    scale_factor = 1
    img1 = cv2.resize(img1, (img1.shape[1] // scale_factor, img1.shape[0] // scale_factor),
                  interpolation = cv2.INTER_AREA)
    torch_img1 = torch.from_numpy(img1/255.).float()[None, None].to(device)

    frame1_data = matching.superpoint({'image': torch_img1})
    frame1_data = {k+'0': frame1_data[k] for k in keys}
    frame1_data['image0'] = torch_img1

    img2 = cv2.resize(img2, (img2.shape[1] // scale_factor, img2.shape[0] // scale_factor),
                interpolation = cv2.INTER_AREA)
    torch_img2 = torch.from_numpy(img2/255.).float()[None, None].to(device)

    pred = matching({**frame1_data, 'image1': torch_img2})
    kpts0 = frame1_data['keypoints0'][0].cpu().numpy()
    kpts1 = pred['keypoints1'][0].cpu().numpy()
    matches = pred['matches0'][0].cpu().numpy()
    confidence = pred['matching_scores0'][0].cpu().numpy()

    valid = matches > -1
    mkpts0 = kpts0[valid]
    mkpts1 = kpts1[matches[valid]]

    threshold = 5
    F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, threshold)
    if inliers is None:
        return 0
    if sum(inliers)[0]>len(mkpts0):
        return 0

    if not is_vis:
        # print(inlier_ratio[0])
        return sum(inliers)[0] # return inlier_ratio

    # 可视化匹配结果和验证结果
    if is_vis:
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(1, 2),
        ]
        out = make_matching_plot_fast(
            image1, image2, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=False, small_text=small_text)
        cv2.imshow('SuperGlue matches', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return sum(inliers)[0]

def evaluate(gt_txt):
    inlier_ratio = []
    labels = []
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference, gt = line_str[0], line_str[1], int(line_str[2])
        inlier_ratio.append(loop_verification(query, reference, new_size=(640,480), is_vis=False, crop=True))
        labels.append(gt)
    return np.array(inlier_ratio), np.array(labels)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            # SuperPoint Non Maximum Suppression (NMS) radius
            'nms_radius': 4,
            # SuperPoint keypoint detector confidence threshold
            'keypoint_threshold': 0.005,
            # Maximum number of keypoints detected by Superpoint
            'max_keypoints': -1
        },
        'superglue': {
            # SuperGlue weights
            'weights': 'outdoor',
            # Number of Sinkhorn iterations performed by SuperGlue
            'sinkhorn_iterations': 20,
            # SuperGlue match threshold
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(device)
    # visualization
    # inliers = loop_verification("Autumn_mini_query/1418133732744920.jpg", "Autumn_mini_query/1418133732869901.jpg", new_size=None, is_vis=True, crop=True)
    # print(inliers)

    # evaluate
    # datasets = ["Kudamm_easy_final.txt", "Kudamm_diff_final.txt"]
    datasets = ["robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt", "robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt"]

    for dataset in datasets:
        print("-------- Processing {} ----------".format(dataset.strip(".txt")))
        inliers, labels = evaluate("datasets/"+dataset)
        inliers = inliers/max(inliers)
        precision, recall, _ = precision_recall_curve(labels, inliers)
        average_precision = average_precision_score(labels, inliers)
        plt.plot(recall, precision, label="{} (AP={:.3f})".format(dataset.strip(".txt"), average_precision))
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Precision-Recall Curves for SuperPoint+SuperGlue")
        plt.savefig("Point_{}.png".format(dataset.strip(".txt")))
        plt.close()
    
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.title("Precision-Recall Curves for SuperPoint+SuperGlue baseline")
    # plt.savefig("pr_curve_SuperPoint+SuperGlue.png")
