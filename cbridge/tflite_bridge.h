#ifndef TFLITE_BRIDGE_H
#define TFLITE_BRIDGE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int num_palm_anchors;  // discovered from model output shape
} BridgeModelInfo;

// Initialize all four TFLite models.
// Returns 0 on success, -1 on failure.
int InitModels(const char* palm_path, const char* landmark_path,
               const char* keypoint_path, const char* point_history_path,
               BridgeModelInfo* info);

// Palm detection inference.
// bgr:  uint8 BGR frame [width * height * 3], any resolution (resized internally)
// out_scores: float32[num_palm_anchors]  raw sigmoid scores
// out_boxes:  float32[num_palm_anchors * 18]  raw box regressors
void RunPalmDetection(const unsigned char* bgr, int width, int height,
                      float* out_scores, float* out_boxes);

// Hand landmark inference.
// bgr: uint8 BGR patch [width * height * 3] (resized to 224x224 internally)
// out_ldmks: float32[63]   21 kps * (x,y,z) in [0..224] patch pixels
// out_hand:  float32[1]    handedness score (>0.5 = right hand)
void RunHandLandmark(const unsigned char* bgr, int width, int height,
                     float* out_ldmks, float* out_hand);

// Keypoint classifier: input float32[42], returns argmax class id.
int RunKeypointClassifier(const float* input, int n);

// Point-history classifier: input float32[32], returns argmax class id.
int RunPointHistoryClassifier(const float* input, int n);

#ifdef __cplusplus
}
#endif
#endif // TFLITE_BRIDGE_H
