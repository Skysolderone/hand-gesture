#include "tflite_bridge.h"
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <cstdio>
#include <cstring>
#include <cmath>

static cv::dnn::Net g_palm;
static cv::dnn::Net g_lm;
static cv::dnn::Net g_kp;
static cv::dnn::Net g_ph;

static int  g_numAnchors = 0;
static int  g_scoresIdx  = 1;
static int  g_boxesIdx   = 0;
static std::vector<std::string> g_palmNames;
static std::vector<std::string> g_lmNames;

static void printShape(const char* label, const cv::Mat& m) {
    fprintf(stderr, "  %-20s dims=%d [", label, m.dims);
    for (int i = 0; i < m.dims; i++)
        fprintf(stderr, "%d%s", m.size[i], i+1 < m.dims ? "," : "");
    fprintf(stderr, "] total=%d\n", (int)m.total());
}

int InitModels(const char* palm_path, const char* landmark_path,
               const char* keypoint_path, const char* point_history_path,
               BridgeModelInfo* info) {
    try {
        g_palm = cv::dnn::readNetFromTFLite(palm_path);
        g_lm   = cv::dnn::readNetFromTFLite(landmark_path);
        g_kp   = cv::dnn::readNetFromTFLite(keypoint_path);
        g_ph   = cv::dnn::readNetFromTFLite(point_history_path);
    } catch (const cv::Exception& e) {
        fprintf(stderr, "[bridge] load error: %s\n", e.what());
        return -1;
    }

    // --- probe palm ---
    // Always use blobFromImage (NCHW [1,C,H,W]); OpenCV DNN handles
    // NCHW→NHWC conversion internally for TFLite models.
    {
        cv::Mat dummy = cv::dnn::blobFromImage(
            cv::Mat::zeros(192, 192, CV_8UC3),
            1.0/127.5, cv::Size(192,192),
            cv::Scalar(127.5,127.5,127.5), true, false, CV_32F);

        g_palmNames = g_palm.getUnconnectedOutLayersNames();
        fprintf(stderr, "[bridge] palm output names (%zu):", g_palmNames.size());
        for (auto& n : g_palmNames) fprintf(stderr, " '%s'", n.c_str());
        fprintf(stderr, "\n");

        g_palm.setInput(dummy, "input_1");
        std::vector<cv::Mat> outs;
        try {
            g_palm.forward(outs, g_palmNames);
        } catch (const cv::Exception& e) {
            fprintf(stderr, "[bridge] palm forward error: %s\n", e.what());
            return -1;
        }

        fprintf(stderr, "[bridge] palm outputs (%zu):\n", outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            printShape(g_palmNames[i].c_str(), outs[i]);

        g_numAnchors = 0;
        for (size_t i = 0; i < outs.size(); i++) {
            if (outs[i].dims < 2) continue;
            int last = outs[i].size[outs[i].dims - 1];
            if (last == 1) {
                g_scoresIdx  = (int)i;
                g_numAnchors = (int)outs[i].total();
            } else if (last == 18) {
                g_boxesIdx = (int)i;
                if (g_numAnchors == 0) g_numAnchors = (int)(outs[i].total() / 18);
            }
        }
        fprintf(stderr, "[bridge] palmAnchors=%d scoresIdx=%d boxesIdx=%d\n",
                g_numAnchors, g_scoresIdx, g_boxesIdx);
        info->num_palm_anchors = g_numAnchors;

        // Print first 8 raw score values (should be logits)
        if (!outs.empty() && g_scoresIdx < (int)outs.size()) {
            const float* sp = outs[g_scoresIdx].ptr<float>();
            int ns = (int)outs[g_scoresIdx].total();
            fprintf(stderr, "[bridge] palm dummy scores[0..7]:");
            for (int i = 0; i < std::min(8, ns); i++)
                fprintf(stderr, " %.3f", sp[i]);
            fprintf(stderr, "\n");
        }
    }

    // --- probe landmark ---
    {
        cv::Mat dummy = cv::dnn::blobFromImage(
            cv::Mat::zeros(224, 224, CV_8UC3),
            1.0/255.0, cv::Size(224,224),
            cv::Scalar(0,0,0), true, false, CV_32F);

        g_lmNames = g_lm.getUnconnectedOutLayersNames();
        g_lm.setInput(dummy, "input_1");
        std::vector<cv::Mat> outs;
        try {
            g_lm.forward(outs, g_lmNames);
        } catch (const cv::Exception& e) {
            fprintf(stderr, "[bridge] landmark forward error: %s\n", e.what());
            return -1;
        }
        fprintf(stderr, "[bridge] landmark outputs (%zu):\n", outs.size());
        for (size_t i = 0; i < outs.size(); i++)
            printShape(g_lmNames[i].c_str(), outs[i]);
    }

    return 0;
}

void RunPalmDetection(const unsigned char* bgr, int width, int height,
                      float* out_scores, float* out_boxes) {
    cv::Mat img(height, width, CV_8UC3, (void*)bgr, (size_t)width * 3);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/127.5, cv::Size(192,192),
                                          cv::Scalar(127.5,127.5,127.5), true, false, CV_32F);
    g_palm.setInput(blob, "input_1");
    std::vector<cv::Mat> outs;
    g_palm.forward(outs, g_palmNames);

    if ((int)outs.size() <= std::max(g_scoresIdx, g_boxesIdx)) return;

    const cv::Mat& scores = outs[g_scoresIdx];
    const cv::Mat& boxes  = outs[g_boxesIdx];

    int ns = std::min((int)scores.total(), g_numAnchors);
    int nb = std::min((int)boxes.total(),  g_numAnchors * 18);
    if (ns > 0) memcpy(out_scores, scores.ptr<float>(), ns * sizeof(float));
    if (nb > 0) memcpy(out_boxes,  boxes.ptr<float>(),  nb * sizeof(float));

    // Debug: print max sigmoid every 60 frames for the first 300, then stop
    static int cnt = 0;
    if (++cnt <= 300 && cnt % 60 == 0) {
        const float* sp = scores.ptr<float>();
        float mx = sp[0];
        int mxi = 0;
        for (int i = 1; i < ns; i++) { if (sp[i] > mx) { mx = sp[i]; mxi = i; } }
        float sig = 1.0f / (1.0f + expf(-mx));
        fprintf(stderr, "[palm] frame=%d maxLogit=%.4f(idx=%d) maxSigmoid=%.4f\n",
                cnt, mx, mxi, sig);
        // Also print top-3 logits values
        fprintf(stderr, "       scores[0..5]: %.4f %.4f %.4f %.4f %.4f %.4f\n",
                sp[0], sp[1], sp[2], sp[3], sp[4], sp[5]);
    }
}

void RunHandLandmark(const unsigned char* bgr, int width, int height,
                     float* out_ldmks, float* out_hand) {
    cv::Mat img(height, width, CV_8UC3, (void*)bgr, (size_t)width * 3);
    cv::Mat blob = cv::dnn::blobFromImage(img, 1.0/255.0, cv::Size(224,224),
                                          cv::Scalar(0,0,0), true, false, CV_32F);
    g_lm.setInput(blob, "input_1");
    std::vector<cv::Mat> outs;
    g_lm.forward(outs, g_lmNames);

    *out_hand = 0.5f;
    for (size_t i = 0; i < outs.size(); i++) {
        int tot = (int)outs[i].total();
        if (tot == 63) {
            memcpy(out_ldmks, outs[i].ptr<float>(), 63 * sizeof(float));
        } else if (tot == 1) {
            *out_hand = outs[i].ptr<float>()[0];
        }
    }
}

static int argmaxF(const float* data, int n) {
    if (n <= 0) return 0;
    int idx = 0;
    for (int i = 1; i < n; i++) if (data[i] > data[idx]) idx = i;
    return idx;
}

int RunKeypointClassifier(const float* input, int n) {
    cv::Mat blob(1, n, CV_32F, (void*)input);
    g_kp.setInput(blob.clone(), "input_1");
    cv::Mat out = g_kp.forward();
    return argmaxF(out.ptr<float>(), (int)out.total());
}

int RunPointHistoryClassifier(const float* input, int n) {
    cv::Mat blob(1, n, CV_32F, (void*)input);
    g_ph.setInput(blob.clone(), "input_1");
    cv::Mat out = g_ph.forward();
    return argmaxF(out.ptr<float>(), (int)out.total());
}
