import string
import weakref
import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import torch

from superglue.matching import Matching
from superglue.utils import (make_matching_plot_fast, frame2tensor)

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines

from sold2.model.line_matcher import LineMatcher
from sold2.misc.visualize_util import plot_images, plot_lines, plot_line_matches, plot_color_line_matches, plot_keypoints

torch.set_grad_enabled(False)

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

    # Detect the lines
    input1 = {'image': torch_img1}
    with torch.no_grad():
        out1 = net(input1)
        line_seg1 = out1['lines'][0][:, :, ::-1]
    with torch.no_grad():
        net_outputs = line_matcher.model(torch_img1)
    des1 = net_outputs["descriptors"]


    input2 = {'image': torch_img2}
    with torch.no_grad():
        out2 = net(input2)
        line_seg2 = out2['lines'][0][:, :, ::-1]
    with torch.no_grad():
        net_outputs = line_matcher.model(torch_img2)
    des2 = net_outputs["descriptors"]

    # outputs = line_matcher([torch_img1, torch_img2])
    # line_seg1 = outputs["line_segments"][0]
    # line_seg2 = outputs["line_segments"][1]
    # matches = outputs["matches"]

    matches = line_matcher.line_matcher.forward(line_seg1, line_seg2, des1, des2)

    valid_matches = matches != -1
    match_indices = matches[valid_matches]
    matched_lines1 = line_seg1[valid_matches][:, :, ::-1]
    matched_lines2 = line_seg2[match_indices][:, :, ::-1]

    # matched_lines1 = matched_lines1.reshape(-1,2)
    # matched_lines2 = matched_lines2.reshape(-1,2)
    points1=[]
    points2=[]
    line_points1, valid_points1 = line_matcher.line_matcher.sample_line_points(matched_lines1)
    line_points1=  line_points1.reshape(-1,2).round().astype(np.int16)
    line_points2, valid_points2 = line_matcher.line_matcher.sample_line_points(matched_lines2)
    line_points2 = line_points2.reshape(-1,2).round().astype(np.int16)
    for i in range(0, len(line_points2)):
        if line_points1[i,0]==0 and  line_points1[i,1]==0:
            continue
        else:
            points1.append(line_points1[i])
            points2.append(line_points2[i])
    points1 = np.int16(points1) 
    points2 = np.int16(points2)

    threshold = 5
    Fp, inliers_p = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.FM_RANSAC, threshold)
    if inliers_p is None:
        return 0
    if sum(inliers_p)[0]>len(mkpts0):
        return 0
    
    Fl, inliers_l = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, threshold)
    if inliers_l is None:
        return 0
    if sum(inliers_l)[0]>len(points1):
        return 0

    threshold = 150

    if not is_vis:
        # here we set the threshold to 150
        if int(sum(inliers_p)[0])+int(sum(inliers_l)[0]) > threshold:
            return 1
        else:
            return 0
        # print(sum(inliers_p)[0])
        # print(sum(inliers_l)[0])
        # return int(sum(inliers_p)[0])+int(sum(inliers_l)[0]) # return inliers

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

        matplotlib.use('tkagg')
        plot_images([image1, image2], ['Image 1 - matched lines', 'Image 2 - matched lines'])
        plot_color_line_matches([matched_lines1, matched_lines2], lw=2)
        plt.show()

        if int(sum(inliers_p)[0])+int(sum(inliers_l)[0]) > threshold:
            return 1
        else:
            return 0

def evaluate(gt_txt):
    inlier_ratio = []
    labels = []
    fp = open(gt_txt, "r")
    fr = open("robotcar_qAutumn_dbNight_val_result.txt", "w")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference = line_str[0], line_str[1].replace('\n','')
        result = loop_verification(query, reference, new_size=(640,480), is_vis=False, crop=True)
        inlier_ratio.append(result)
        fr.write(str(result)+"\n")

    return np.array(inlier_ratio)

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

    # SOLD2
    ckpt_path = 'SOLD2/pretrained_models/sold2_wireframe.tar'
    device = 'cuda'
    mode = 'dynamic'  # 'dynamic' or 'static'

    config = {
        'model_cfg': {
            'model_name': "lcnn_simple",
            'model_architecture': "simple",
            # Backbone related config
            'backbone': "lcnn",
            'backbone_cfg': {
                'input_channel': 1, # Use RGB images or grayscale images.
                'depth': 4,
                'num_stacks': 2,
                'num_blocks': 1,
                'num_classes': 5
            },
            # Junction decoder related config
            'junction_decoder': "superpoint_decoder",
            'junc_decoder_cfg': {},
            # Heatmap decoder related config
            'heatmap_decoder': "pixel_shuffle",
            'heatmap_decoder_cfg': {},
            # Descriptor decoder related config
            'descriptor_decoder': "superpoint_descriptor",
            'descriptor_decoder_cfg': {},
            # Shared configurations
            'grid_size': 8,
            'keep_border_valid': True,
            # Threshold of junction detection
            'detection_thresh': 0.0153846, # 1/65
            'max_num_junctions': 300,
            # Threshold of heatmap detection
            'prob_thresh': 0.5,
            # Weighting related parameters
            'weighting_policy': mode,
            # [Heatmap loss]
            'w_heatmap': 0.,
            'w_heatmap_class': 1,
            'heatmap_loss_func': "cross_entropy",
            'heatmap_loss_cfg': {
                'policy': mode
            },
            # [Heatmap consistency loss]
            # [Junction loss]
            'w_junc': 0.,
            'junction_loss_func': "superpoint",
            'junction_loss_cfg': {
                'policy': mode
            },
            # [Descriptor loss]
            'w_desc': 0.,
            'descriptor_loss_func': "regular_sampling",
            'descriptor_loss_cfg': {
                'dist_threshold': 8,
                'grid_size': 4,
                'margin': 1,
                'policy': mode
            },
        },
        'line_detector_cfg': {
            'detect_thresh': 0.15,  # depending on your images, you might need to tune this parameter
            'num_samples': 64,
            'sampling_method': "local_max",
            'inlier_thresh': 0.9,
            "use_candidate_suppression": True,
            "nms_dist_tolerance": 3.,
            "use_heatmap_refinement": True,
            "heatmap_refine_cfg": {
                "mode": "local",
                "ratio": 0.2,
                "valid_thresh": 1e-3,
                "num_blocks": 20,
                "overlap_ratio": 0.5
            }
        },
        'multiscale': False,
        'line_matcher_cfg': {
            'cross_check': True,
            'num_samples': 5,
            'min_dist_pts': 8,
            'top_k_candidates': 10,
            'grid_size': 4
        }
    }

    line_matcher = LineMatcher(
            config["model_cfg"], ckpt_path, device, config["line_detector_cfg"],
            config["line_matcher_cfg"], config["multiscale"])

    
    # Model config for DeepLSD
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': False,  # Whether to merge close-by lines
            'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 3,
            'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    # Load the DeepLSD model
    ckpt = 'DeepLSD/weights/deeplsd_md.tar'
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()

    # visualization
    # result = loop_verification("Autumn_mini_query/1418133732744920.jpg", "Autumn_mini_query/1418133732869901.jpg", new_size=None, is_vis=False, crop=True)
    # print(result)

    # evaluate
    datasets = ["robotcar_qAutumn_dbNight_val_final.txt"]

    for dataset in datasets:
        print("-------- Processing {} ----------".format(dataset.strip(".txt")))
        inliers = evaluate("datasets/"+dataset)