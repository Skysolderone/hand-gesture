package cbridge

// #cgo CXXFLAGS: -std=c++17
// #cgo pkg-config: opencv4
// #include "tflite_bridge.h"
// #include <stdlib.h>
import "C"
import (
	"fmt"
	"unsafe"
)

// ModelInfo is returned from Init.
type ModelInfo struct {
	NumPalmAnchors int
}

// Init loads all four TFLite models.
func Init(palmPath, landmarkPath, keypointPath, pointHistoryPath string) (ModelInfo, error) {
	cp := C.CString(palmPath)
	cl := C.CString(landmarkPath)
	ck := C.CString(keypointPath)
	cph := C.CString(pointHistoryPath)
	defer C.free(unsafe.Pointer(cp))
	defer C.free(unsafe.Pointer(cl))
	defer C.free(unsafe.Pointer(ck))
	defer C.free(unsafe.Pointer(cph))

	var info C.BridgeModelInfo
	if ret := C.InitModels(cp, cl, ck, cph, &info); ret != 0 {
		return ModelInfo{}, fmt.Errorf("InitModels returned %d", ret)
	}
	return ModelInfo{NumPalmAnchors: int(info.num_palm_anchors)}, nil
}

// RunPalmDetection runs palm detection on a BGR uint8 frame.
// bgr must be width*height*3 bytes.
// Returns scores [numAnchors] and boxes [numAnchors*18].
func RunPalmDetection(bgr []byte, width, height, numAnchors int) (scores []float32, boxes []float32) {
	scores = make([]float32, numAnchors)
	boxes = make([]float32, numAnchors*18)
	C.RunPalmDetection(
		(*C.uchar)(unsafe.Pointer(&bgr[0])),
		C.int(width), C.int(height),
		(*C.float)(unsafe.Pointer(&scores[0])),
		(*C.float)(unsafe.Pointer(&boxes[0])),
	)
	return
}

// RunHandLandmark runs hand landmark detection on a BGR uint8 patch.
// Returns 63 landmark floats (21 kps × xyz in [0..224]) and handedness score.
func RunHandLandmark(bgr []byte, width, height int) ([63]float32, float32) {
	var ldmks [63]C.float
	var hand C.float
	C.RunHandLandmark(
		(*C.uchar)(unsafe.Pointer(&bgr[0])),
		C.int(width), C.int(height),
		&ldmks[0], &hand,
	)
	var res [63]float32
	for i := 0; i < 63; i++ {
		res[i] = float32(ldmks[i])
	}
	return res, float32(hand)
}

// RunKeypointClassifier classifies 42 normalised landmark floats.
func RunKeypointClassifier(input []float32) int {
	return int(C.RunKeypointClassifier(
		(*C.float)(unsafe.Pointer(&input[0])),
		C.int(len(input)),
	))
}

// RunPointHistoryClassifier classifies 32 floats (16-frame history).
func RunPointHistoryClassifier(input []float32) int {
	return int(C.RunPointHistoryClassifier(
		(*C.float)(unsafe.Pointer(&input[0])),
		C.int(len(input)),
	))
}
