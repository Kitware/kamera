import argparse
import cv2
import faiss
import os
import pathlib
import time
import pycolmap as pc
import numpy as np
# assuming GPU pycolmap and faiss are installed for optimal speed

sift_options = pc.SiftExtractionOptions()
sift_options.max_num_features = 2048
sift_options.max_image_size = 1280
sift = pc.Sift(sift_options)
rsz = (1280, 720)  # w, h


def colmap_sift_extract(frame) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = np.array(img).astype(np.float32) / 255.0
    kpts, feat = sift.extract(img)
    pts2d = kpts[:, :2]  # Just need x and y
    feat = feat / np.linalg.norm(feat, axis=1, keepdims=True)
    return pts2d, feat


def match_descriptors(
    descriptors: np.ndarray, faiss_index: faiss.Index, ratio_thresh: float = 0.7
) -> list:
    # Find 2 nearest neighbors for each descriptor
    descriptors = np.vstack(descriptors, dtype=np.float32)
    distances, indices = faiss_index.search(descriptors, 2)
    # Apply Lowe's ratio test
    good_matches = []
    for i, (d1, d2) in enumerate(distances):
        if d1 < ratio_thresh * d2:
            match = cv2.DMatch(_queryIdx=i, _trainIdx=indices[i][0], _distance=d1)
            good_matches.append(match)
    return good_matches


def calculate_overlap(keypoints1: np.ndarray,
                      keypoints2: np.ndarray,
                      frame1: cv2.typing.MatLike,
                      frame2: cv2.typing.MatLike,
                      matches: list):
    if len(matches) < 10:
        # not enough matches, assume 0 overlap
        return 0.0
    ptsA = np.float32([keypoints1[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
    ptsB = np.float32([keypoints2[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 4.0)  # B -> A
    if H is None:
        return 0.0

    hA, wA = frame1.shape[:2]
    hB, wB = frame2.shape[:2]

    rectA = np.float32([[0, 0], [wA, 0], [wA, hA], [0, hA]])
    rectB = np.float32([[0, 0], [wB, 0], [wB, hB], [0, hB]]).reshape(-1, 1, 2)
    warpedB = cv2.perspectiveTransform(rectB, H).reshape(-1, 2)

    inter_area, _ = cv2.intersectConvexConvex(rectA, warpedB)
    overlap_frac = max(0.0, min(1.0, inter_area / float(wA * hA)))  # overlap wrt A

    return overlap_frac


def set_new_kf(
    frame: cv2.typing.MatLike,
    rz_frame: cv2.typing.MatLike,
    idx: int,
    outdir: os.PathLike | str,
    faiss_index: faiss.Index,
    descriptors: np.ndarray,
    keypoints: np.ndarray,
    prefix: str,
) -> dict:
    # set new KF and return a dict of data
    out_path = os.path.join(outdir, f"{prefix}_frame_{idx:06d}.jpg")
    cv2.imwrite(out_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    index_descriptors = np.vstack(descriptors, dtype=np.float32)
    faiss_index.reset()
    faiss_index.add(index_descriptors)
    kf_data = {
        "frame": frame,
        "rz_frame": rz_frame,
        "descriptors": index_descriptors,
        "keypoints": keypoints,
        "idx": idx,
    }
    return kf_data


def main():
    ap = argparse.ArgumentParser(description="Save frames from a video as JPEGs based on a set target overlap.")
    ap.add_argument("video", help="Path to input .mp4")
    args = ap.parse_args()

    vpath = pathlib.Path(args.video)
    prefix = vpath.stem
    outdir = f"./images/{vpath.stem}"

    print(f"Writing images to {outdir}.")
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    VECTOR_DIM = 128
    DESIRED_OVERLAP = 0.7
    # Build index for keyframes
    kf_faiss = faiss.IndexFlatL2(VECTOR_DIM)
    use_gpu = True
    if use_gpu:
        res = faiss.StandardGpuResources()
        kf_faiss = faiss.index_cpu_to_gpu(res, 0, kf_faiss)

    idx = saved = 0
    prev_frame_data = kf_data = None
    while True:
        tic = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        rz_frame = cv2.resize(frame, rsz)
        frame_keypoints, frame_descriptors = colmap_sift_extract(rz_frame)

        if idx == 0:
            kf_data = set_new_kf(
                frame,
                rz_frame,
                idx,
                outdir,
                kf_faiss,
                frame_descriptors,
                frame_keypoints,
                prefix,
            )
            saved += 1
        else:
            matches = match_descriptors(frame_descriptors, kf_faiss)
            overlap = calculate_overlap(
                frame_keypoints,
                kf_data["keypoints"],
                rz_frame,
                kf_data["rz_frame"],
                matches,
            )
            if overlap < DESIRED_OVERLAP:
                # if the previous frame was a keyframe, just set the current as
                # a keyframe and move on
                if (idx - 1) == kf_data["idx"]:
                    kf_data = set_new_kf(
                        frame,
                        rz_frame,
                        idx,
                        outdir,
                        kf_faiss,
                        frame_descriptors,
                        frame_keypoints,
                        prefix,
                    )
                    saved += 1
                else:
                    # otherwise, set previous frame as a new key frame and
                    # re-compute the overlap against that frame
                    kf_data = set_new_kf(
                        prev_frame_data["frame"],
                        prev_frame_data["rz_frame"],
                        (idx - 1),
                        outdir,
                        kf_faiss,
                        prev_frame_data["descriptors"],
                        prev_frame_data["keypoints"],
                        prefix,
                    )
                    saved += 1
                    matches = match_descriptors(frame_descriptors, kf_faiss)
                    overlap = calculate_overlap(
                        frame_keypoints,
                        kf_data["keypoints"],
                        rz_frame,
                        kf_data["rz_frame"],
                        matches,
                    )
                    if overlap < DESIRED_OVERLAP:
                        # if it's still not within our bounds, move the keyframe pointer
                        # to the current frame
                        kf_data = set_new_kf(
                            frame,
                            rz_frame,
                            idx,
                            outdir,
                            kf_faiss,
                            frame_descriptors,
                            frame_keypoints,
                            prefix,
                        )
                        saved += 1

        prev_frame_data = {
            "frame": frame,
            "rz_frame": rz_frame,
            "descriptors": frame_descriptors,
            "keypoints": frame_keypoints,
        }
        idx += 1
        toc = time.time()
        print("Time to process frame was %0.3fs." % (toc - tic))

    cap.release()
    print(f"Saved {saved} / {idx} frame(s) to {outdir}.")


if __name__ == "__main__":
    main()
