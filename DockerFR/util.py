"""
Utility implementations for the face recognition project.

- Face detection & keypoints: RetinaFace (retina-face package)
- Alignment: 5-point landmarks -> 112x112 RetinaFace template
- Embedding: ONNX ArcFace pretrained model (high performance)
- Similarity: cosine similarity between L2-normalized embeddings
"""

from typing import Any, List, Dict
import cv2
import numpy as np
import torch
import torch.nn as nn
from retinaface import RetinaFace
import onnxruntime as ort


# ------------------------------
# RetinaFace 5-point reference template (112x112)
# ------------------------------
REFERENCE_FACIAL_POINTS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

ALIGNED_FACE_SIZE = (112, 112)


# ------------------------------
# ONNX Embedding Model Loader
# ------------------------------
_ONNX_SESSION = None

def _get_onnx_embedder():
    """
    Lazy load ONNX ArcFace model.
    Change the model path below if needed.
    """
    global _ONNX_SESSION
    if _ONNX_SESSION is None:
        _ONNX_SESSION = ort.InferenceSession(
            "models/model.onnx",
            providers=["CPUExecutionProvider"]
        )
    return _ONNX_SESSION


# ------------------------------
# Helper functions
# ------------------------------

def _decode_image(image: Any) -> np.ndarray:
    """
    Accept raw bytes or a decoded numpy BGR image and return BGR np.ndarray.
    """
    if isinstance(image, (bytes, bytearray)):
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from byte buffer.")
        return img
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def _select_largest_face(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Among detected faces, select the one with the largest bounding box area.
    """
    if not faces:
        raise ValueError("No faces provided.")
    return max(
        faces,
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
    )


def _estimate_affine_matrix(
    src_landmarks: np.ndarray,
    dst_landmarks: np.ndarray = REFERENCE_FACIAL_POINTS,
) -> np.ndarray:
    """
    Estimate 2x3 affine transform mapping src_landmarks -> dst_landmarks.
    """
    src = src_landmarks.astype(np.float32)
    dst = dst_landmarks.astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("Failed to estimate affine transform for face alignment.")
    return matrix


# ------------------------------
# 1. Face Detection
# ------------------------------

def detect_faces(image: Any) -> List[Dict[str, Any]]:
    """
    Detect faces within the provided image using RetinaFace.
    Return bbox, landmarks, and face crops.
    """
    img_bgr = _decode_image(image)
    resp = RetinaFace.detect_faces(img_bgr)

    faces: List[Dict[str, Any]] = []
    if not isinstance(resp, dict):
        return faces

    for _, data in resp.items():
        x1, y1, x2, y2 = data["facial_area"]
        x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
        x1i = max(x1i, 0)
        y1i = max(y1i, 0)
        x2i = min(x2i, img_bgr.shape[1])
        y2i = min(y2i, img_bgr.shape[0])

        face_crop = img_bgr[y1i:y2i, x1i:x2i].copy()

        lm = data.get("landmarks", {})
        landmarks_abs = np.array(
            [
                lm["left_eye"],
                lm["right_eye"],
                lm["nose"],
                lm["mouth_left"],
                lm["mouth_right"],
            ],
            dtype=np.float32,
        )
        landmarks_local = landmarks_abs.copy()
        landmarks_local[:, 0] -= x1i
        landmarks_local[:, 1] -= y1i

        faces.append(
            {
                "bbox": np.array([x1i, y1i, x2i, y2i], dtype=np.float32),
                "landmarks": landmarks_local,
                "face_image": face_crop,
            }
        )

    return faces


# ------------------------------
# 2. Keypoint Detection
# ------------------------------

def detect_face_keypoints(face_image: Any) -> Any:
    """
    Return 5 key landmarks.
    If already provided by detect_faces, reuse them.
    """
    if isinstance(face_image, dict) and "landmarks" in face_image:
        return face_image["landmarks"]

    faces = detect_faces(face_image)
    if not faces:
        raise ValueError("No face detected while detecting keypoints.")
    largest = _select_largest_face(faces)
    return largest["landmarks"]


# ------------------------------
# 3. Warping / Alignment
# ------------------------------

def warp_face(image: Any, homography_matrix: Any) -> Any:
    """
    Warp face using 2x3 affine transform to 112×112.
    """
    img_bgr = _decode_image(image)
    M = np.asarray(homography_matrix, dtype=np.float32)

    if M.shape == (2, 3):
        aligned = cv2.warpAffine(img_bgr, M, ALIGNED_FACE_SIZE)
    elif M.shape == (3, 3):
        aligned = cv2.warpPerspective(img_bgr, M, ALIGNED_FACE_SIZE)
    else:
        raise ValueError(f"Invalid matrix shape: {M.shape}")
    return aligned


# ------------------------------
# 4. Anti-spoofing (simple heuristic)
# ------------------------------

def antispoof_check(face_image: Any) -> float:
    """
    Simple Laplacian variance-based spoof check (0~1).
    """
    img_bgr = _decode_image(face_image)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = lap_var / 1000.0
    return float(max(0.0, min(score, 1.0)))


# ------------------------------
# 5. Embedding (ONNX ArcFace)
# ------------------------------

def compute_face_embedding(face_image: Any) -> Any:
    """
    Compute 512-D embedding using ONNX ArcFace model.
    Input: aligned 112×112 BGR
    """
    img_bgr = _decode_image(face_image)
    img_bgr = cv2.resize(img_bgr, ALIGNED_FACE_SIZE)

    # BGR → RGB, [0,1]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # HWC → NCHW
    blob = np.transpose(img_rgb, (2, 0, 1))[None, :, :, :]

    session = _get_onnx_embedder()
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    embedding = session.run([output_name], {input_name: blob})[0]
    embedding = embedding.squeeze().astype(np.float32)

    # L2 normalize
    norm = np.linalg.norm(embedding) + 1e-10
    embedding = embedding / norm

    return embedding


# ------------------------------
# 6. Full Similarity Pipeline
# ------------------------------

def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
    """
    Full end-to-end similarity pipeline.
    """
    img_a = _decode_image(image_a)
    img_b = _decode_image(image_b)

    faces_a = detect_faces(img_a)
    faces_b = detect_faces(img_b)

    if not faces_a:
        raise ValueError("No face detected in image_a.")
    if not faces_b:
        raise ValueError("No face detected in image_b.")

    face_a = _select_largest_face(faces_a)
    face_b = _select_largest_face(faces_b)

    # Keypoints
    lmk_a = detect_face_keypoints(face_a)
    lmk_b = detect_face_keypoints(face_b)

    # Affine matrix (align)
    M_a = _estimate_affine_matrix(lmk_a)
    M_b = _estimate_affine_matrix(lmk_b)

    aligned_a = warp_face(face_a["face_image"], M_a)
    aligned_b = warp_face(face_b["face_image"], M_b)

    # Spoof check
    spoof_a = antispoof_check(aligned_a)
    spoof_b = antispoof_check(aligned_b)

    if spoof_a < 0.25 or spoof_b < 0.25:
        raise ValueError("Spoof detected: one of the faces seems fake.")

    # Embedding
    emb_a = compute_face_embedding(aligned_a)
    emb_b = compute_face_embedding(aligned_b)

    # Cosine similarity (already normalized)
    similarity = float(np.dot(emb_a, emb_b))
    return max(-1.0, min(1.0, similarity))


# """
# Utility implementations for the face recognition project.

# - Face detection & keypoints: RetinaFace (retina-face package)
# - Alignment: 5-point landmarks -> 112x112 RetinaFace template
# - Embedding: torchvision ResNet18 (pretrained on ImageNet, 512-D features)
# - Similarity: cosine similarity between L2-normalized embeddings
# """

# from typing import Any, List, Dict

# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# from torchvision import models
# from retinaface import RetinaFace


# # ------------------------------
# # RetinaFace 5-point reference template (112x112)
# # (RetinaFace 논문 및 InsightFace 구현에서 사용되는 좌표)
# # ------------------------------
# REFERENCE_FACIAL_POINTS = np.array(
#     [
#         [38.2946, 51.6963],  # left eye
#         [73.5318, 51.5014],  # right eye
#         [56.0252, 71.7366],  # nose tip
#         [41.5493, 92.3655],  # left mouth corner
#         [70.7299, 92.2041],  # right mouth corner
#     ],
#     dtype=np.float32,
# )

# ALIGNED_FACE_SIZE = (112, 112)


# # ------------------------------
# # Embedding model (ResNet18 backbone)
# # ------------------------------

# _EMBEDDER: nn.Module | None = None


# def _get_embedder() -> nn.Module:
#     """
#     Lazy-load a ResNet18 pretrained on ImageNet and use the 512-D features
#     as a face embedding vector.
#     """
#     global _EMBEDDER
#     if _EMBEDDER is None:
#         # torchvision >=0.13 style (weights API)
#         try:
#             weights = models.ResNet18_Weights.IMAGENET1K_V1
#             resnet = models.resnet18(weights=weights)
#         except AttributeError:
#             # older torchvision fallback
#             resnet = models.resnet18(pretrained=True)

#         # Replace the classification head with identity to get 512-D features
#         resnet.fc = nn.Identity()
#         resnet.eval()
#         _EMBEDDER = resnet
#     return _EMBEDDER


# # ------------------------------
# # Helper functions
# # ------------------------------

# def _decode_image(image: Any) -> np.ndarray:
#     """
#     Accept raw bytes or an already-decoded numpy image (BGR) and return BGR np.ndarray.
#     """
#     if isinstance(image, (bytes, bytearray)):
#         nparr = np.frombuffer(image, np.uint8)
#         img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#         if img is None:
#             raise ValueError("Could not decode image from byte buffer.")
#         return img
#     if isinstance(image, np.ndarray):
#         return image
#     raise TypeError(f"Unsupported image type: {type(image)}")


# def _select_largest_face(faces: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """
#     Among detected faces, select the one with the largest bounding box area.
#     Each face dict must contain a key 'bbox' = [x1,y1,x2,y2].
#     """
#     if not faces:
#         raise ValueError("No faces provided.")
#     return max(
#         faces,
#         key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
#     )


# def _estimate_affine_matrix(
#     src_landmarks: np.ndarray,
#     dst_landmarks: np.ndarray = REFERENCE_FACIAL_POINTS,
# ) -> np.ndarray:
#     """
#     Estimate 2x3 affine transform that maps src_landmarks -> dst_landmarks.
#     """
#     src = src_landmarks.astype(np.float32)
#     dst = dst_landmarks.astype(np.float32)
#     # use partial affine (scale + rotation + translation)
#     matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
#     if matrix is None:
#         raise ValueError("Failed to estimate affine transform for face alignment.")
#     return matrix


# # ------------------------------
# # 1. Face Detection
# # ------------------------------

# def detect_faces(image: Any) -> List[Dict[str, Any]]:
#     """
#     Detect faces within the provided image using RetinaFace.

#     Parameters
#     ----------
#     image : bytes or np.ndarray
#         Raw image bytes (as provided by FastAPI) or a decoded BGR image.

#     Returns
#     -------
#     faces : list of dict
#         Each dict has keys:
#           - 'bbox': np.ndarray [x1, y1, x2, y2]
#           - 'landmarks': np.ndarray of shape (5, 2) [x, y] in *crop-local* coords
#           - 'face_image': cropped face image (BGR)
#     """
#     img_bgr = _decode_image(image)

#     # retina-face can take either path or np.ndarray; here we pass the array directly.
#     resp = RetinaFace.detect_faces(img_bgr)

#     faces: List[Dict[str, Any]] = []
#     if not isinstance(resp, dict):
#         return faces  # no faces found

#     for _, data in resp.items():
#         x1, y1, x2, y2 = data["facial_area"]
#         # ensure int for slicing
#         x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#         x1i = max(x1i, 0)
#         y1i = max(y1i, 0)
#         x2i = min(x2i, img_bgr.shape[1])
#         y2i = min(y2i, img_bgr.shape[0])

#         face_crop = img_bgr[y1i:y2i, x1i:x2i].copy()

#         lm = data.get("landmarks", {})
#         # landmarks are given in absolute image coordinates; convert to crop-local
#         landmarks_abs = np.array(
#             [
#                 lm["left_eye"],
#                 lm["right_eye"],
#                 lm["nose"],
#                 lm["mouth_left"],
#                 lm["mouth_right"],
#             ],
#             dtype=np.float32,
#         )
#         landmarks_local = landmarks_abs.copy()
#         landmarks_local[:, 0] -= x1i
#         landmarks_local[:, 1] -= y1i

#         faces.append(
#             {
#                 "bbox": np.array([x1i, y1i, x2i, y2i], dtype=np.float32),
#                 "landmarks": landmarks_local,
#                 "face_image": face_crop,
#             }
#         )

#     return faces


# # ------------------------------
# # 2. Keypoint Detection
# # ------------------------------

# def detect_face_keypoints(face_image: Any) -> Any:
#     """
#     Identify facial keypoints (landmarks) for alignment or analysis.

#     - If `face_image` is a dict produced by `detect_faces`, return its landmarks.
#     - Otherwise, run RetinaFace again on the given image and return the
#       landmarks of the largest detected face.
#     """
#     # Case 1: dict from detect_faces
#     if isinstance(face_image, dict) and "landmarks" in face_image:
#         return face_image["landmarks"]

#     # Case 2: raw bytes or np.ndarray -> run detection again
#     faces = detect_faces(face_image)
#     if not faces:
#         raise ValueError("No face detected while trying to detect keypoints.")
#     largest = _select_largest_face(faces)
#     return largest["landmarks"]


# # ------------------------------
# # 3. Warping / Alignment
# # ------------------------------

# def warp_face(image: Any, homography_matrix: Any) -> Any:
#     """
#     Warp the provided face image using the supplied homography/affine matrix.

#     For this project we use a 2x3 affine transform and normalize to 112x112.
#     """
#     img_bgr = _decode_image(image)
#     M = np.asarray(homography_matrix, dtype=np.float32)

#     if M.shape == (2, 3):
#         aligned = cv2.warpAffine(img_bgr, M, ALIGNED_FACE_SIZE)
#     elif M.shape == (3, 3):
#         aligned = cv2.warpPerspective(img_bgr, M, ALIGNED_FACE_SIZE)
#     else:
#         raise ValueError(f"Unsupported homography/affine matrix shape: {M.shape}")
#     return aligned


# # ------------------------------
# # 4. Anti-spoofing (very simple heuristic)
# # ------------------------------

# def antispoof_check(face_image: Any) -> float:
#     """
#     Perform a very simple anti-spoofing check using image sharpness.

#     This is NOT a production-quality spoof detector, but it provides a
#     reasonable score:
#       - higher Laplacian variance -> sharper -> more likely to be real
#     Returns a score in [0, 1].
#     """
#     img_bgr = _decode_image(face_image)
#     gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

#     # Normalize variance to [0,1] with a simple heuristic
#     score = lap_var / 1000.0
#     score = max(0.0, min(float(score), 1.0))
#     return score


# # ------------------------------
# # 5. Embedding
# # ------------------------------

# def compute_face_embedding(face_image: Any) -> Any:
#     """
#     Compute a numerical embedding vector for the provided face image.

#     - Input is assumed to be an aligned face (112x112 BGR) or raw bytes.
#     - Output is a L2-normalized 512-D numpy array.
#     """
#     img_bgr = _decode_image(face_image)
#     img_bgr = cv2.resize(img_bgr, ALIGNED_FACE_SIZE)

#     # Convert BGR -> RGB, scale to [0,1]
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

#     # Standard normalization similar to ImageNet models (optional, but common)
#     # Here we just center to 0-mean for simplicity.
#     img_rgb = (img_rgb - 0.5) / 0.5

#     # HWC -> CHW for PyTorch
#     tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

#     embedder = _get_embedder()
#     with torch.no_grad():
#         features = embedder(tensor)  # shape: [1, 512]
#     emb = features.squeeze(0).cpu().numpy().astype(np.float32)

#     # L2 normalization
#     norm = np.linalg.norm(emb) + 1e-10
#     emb = emb / norm
#     return emb


# # ------------------------------
# # 6. Full Pipeline: Similarity
# # ------------------------------

# def calculate_face_similarity(image_a: Any, image_b: Any) -> float:
#     """
#     End-to-end pipeline that returns a similarity score between two faces.

#     Steps:
#       1. Decode input bytes to images.
#       2. Detect faces (RetinaFace) and pick the largest one in each image.
#       3. Align faces to 112x112 using 5-point landmarks and affine transform.
#       4. (Optionally) run anti-spoofing checks (we compute but do not threshold).
#       5. Generate embeddings via ResNet18 feature extractor.
#       6. Return cosine similarity between embeddings in [-1, 1].
#     """
#     # 1. Decode (mainly to validate input)
#     img_a = _decode_image(image_a)
#     img_b = _decode_image(image_b)

#     # 2. Detect faces
#     faces_a = detect_faces(img_a)
#     faces_b = detect_faces(img_b)

#     if not faces_a:
#         raise ValueError("No face detected in image_a.")
#     if not faces_b:
#         raise ValueError("No face detected in image_b.")

#     face_a = _select_largest_face(faces_a)
#     face_b = _select_largest_face(faces_b)

#     # 3. Get landmarks and alignment
#     landmarks_a = detect_face_keypoints(face_a)  # already crop-local
#     landmarks_b = detect_face_keypoints(face_b)

#     M_a = _estimate_affine_matrix(landmarks_a, REFERENCE_FACIAL_POINTS)
#     M_b = _estimate_affine_matrix(landmarks_b, REFERENCE_FACIAL_POINTS)

#     aligned_a = warp_face(face_a["face_image"], M_a)
#     aligned_b = warp_face(face_b["face_image"], M_b)

#     # 4. Anti-spoofing (currently not used to reject, just computed)
#     _ = antispoof_check(aligned_a)
#     _ = antispoof_check(aligned_b)

#     # 5. Embeddings
#     emb_a = compute_face_embedding(aligned_a)
#     emb_b = compute_face_embedding(aligned_b)

#     # 6. Cosine similarity (embeddings already L2-normalized)
#     similarity = float(np.dot(emb_a, emb_b))

#     # Numerical safety: clamp to [-1, 1]
#     similarity = max(-1.0, min(1.0, similarity))
#     return similarity
