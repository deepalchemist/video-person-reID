from __future__ import print_function, absolute_import
import numpy as np
import os.path as osp
from tqdm import tqdm

import lib.utils.utils as util


def evaluate(distmat,
             q_pids, g_pids,
             q_camids, g_camids,
             q_paths=None, g_paths=None,
             plot_ranking=False,
             max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    if plot_ranking:
        rank_result_dir = './cache/ranking/mars/'
        util.mkdir(rank_result_dir)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.
    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        # ---------------------- plot ranking results ------------------------
        if plot_ranking:
            assert q_paths is not None and g_paths is not None
            g_paths = np.asarray(g_paths)
            top10 = g_paths[indices[q_idx]][keep][:10]
            top10_ids = g_pids[indices[q_idx]][keep][:10].tolist()

            if top10_ids[0] != q_pids[q_idx]:  # only plot ranking list of error top1
                util.save_vid_rank_result(q_paths[q_idx], top10,
                                          save_path=osp.join(rank_result_dir, osp.basename(q_paths[q_idx][0])))

                # save ground truth ranking list
                # TODO(NOTE): same id and different camera
                ground_truth = ((g_pids[indices[q_idx]] == q_pids[q_idx]) &
                                (g_camids[indices[q_idx]] != q_camids[q_idx]))
                ground_truth = np.where(ground_truth == 1)[0]
                top10 = g_paths[indices[q_idx]][ground_truth][:10]
                util.save_vid_rank_result(q_paths[q_idx], top10,
                                          save_path=osp.join(rank_result_dir,
                                                             osp.basename(q_paths[q_idx][0]).split('.')[
                                                                 0] + '_gt.jpg'))
        # ---------------------------------------------------------------------

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
