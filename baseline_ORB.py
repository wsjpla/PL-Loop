import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def loop_verification(img_fname_1, img_fname_2, new_size=None, is_vis=False, crop=True):
    # 读取两幅图像
    image1 = cv2.imread(img_fname_1, 0)
    if new_size is not None:
        image1 = cv2.resize(image1, new_size)
    image2 = cv2.imread(img_fname_2, 0)
    if new_size is not None:
        image2 = cv2.resize(image2, new_size)

    if crop == True:
        img1 = image1[0:400, :]
        img2 = image2[0:400, :]
    else:
        img1 = image1
        img2 = image2

    # 提取关键点和描述符
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    points1=[]
    points2=[]
    good_matches = []

    # 进行特征匹配并进行对极几何约束验证
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
 
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance

    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_matches.append(x)
            points1.append(kp1[x.queryIdx].pt)
            points2.append(kp2[x.trainIdx].pt) 
    
    # if len(good_matches) == 0:
    #     return 0
    points1 = np.int32(points1) 
    points2 = np.int32(points2)
    threshold = 5
    F, inliers = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, threshold,)
    if inliers is None:
        return 0
    if sum(inliers)[0]>len(good_matches):
        return 0

    if not is_vis:
        # print(inlier_ratio[0])
        return sum(inliers)[0]  # return inlier_ratio
        # return len(good_matches)

    # if len(good_matches) > 30: # 判断是否存在闭环
    #     if is_vis:
    #         print('Found loop closure!')
    #     else:
    #         return 1
    # else:
    #     if is_vis:
    #         print('No loop closure found.')
    #     else:
    #         return 0

    # 可视化匹配结果和验证结果
    if is_vis:
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img3)
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
        inlier_ratio.append(loop_verification(query, reference, new_size=None, is_vis=False))
        labels.append(gt)
    return np.array(inlier_ratio), np.array(labels)


if __name__ == '__main__':
    # visualization
    # inliers = loop_verification("Autumn_mini_query/1418133732744920.jpg", "Autumn_mini_query/1418133732869901.jpg", new_size=None, is_vis=True)
    # print(inliers)

    # evaluate
    # datasets = ["Kudamm_easy_final.txt", "Kudamm_diff_final.txt", "robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt", "robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt"]
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
        plt.title("Precision-Recall Curves for ORB baseline")
        plt.savefig("ORB_{}.png".format(dataset.strip(".txt")))
        plt.close()
    
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend()
    # plt.title("Precision-Recall Curves for ORB baseline")
    # plt.savefig("pr_curve_ORB.png")








    
