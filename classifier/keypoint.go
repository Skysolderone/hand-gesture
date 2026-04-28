package classifier

import "v1/cbridge"

// ClassifyKeypoint returns the gesture class ID for the pre-processed
// 42-float landmark vector.
func ClassifyKeypoint(input []float32) int {
	if len(input) < 42 {
		return 0
	}
	return cbridge.RunKeypointClassifier(input[:42])
}
