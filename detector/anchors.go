package detector

// Anchor represents a prior anchor box center (all anchors have W=H=1.0 in
// palm_detection_lite because fixed_anchor_size=true).
type Anchor struct {
	CX, CY float32
}

// GeneratePalmAnchors generates the 2016 anchors for palm_detection_lite.tflite.
//
// SsdAnchorsCalculatorOptions:
//
//	strides               = [8, 16]
//	input_size            = 192
//	reduce_boxes_in_lowest_layer = true  → 3 anchors/location for stride 8
//	interpolated_scale_aspect_ratio = 1.0 → 2 anchors/location for stride 16
//	fixed_anchor_size     = true
//
// Total: 24×24×3 + 12×12×2 = 1728 + 288 = 2016
func GeneratePalmAnchors() []Anchor {
	const inputSize = 192
	type layerCfg struct {
		stride   int
		nAnchors int // anchors per (x,y) location
	}
	layers := []layerCfg{
		{8, 3},
		{16, 2},
	}
	var anchors []Anchor
	for _, l := range layers {
		fm := inputSize / l.stride
		for y := 0; y < fm; y++ {
			for x := 0; x < fm; x++ {
				cx := (float32(x) + 0.5) / float32(fm)
				cy := (float32(y) + 0.5) / float32(fm)
				for a := 0; a < l.nAnchors; a++ {
					anchors = append(anchors, Anchor{CX: cx, CY: cy})
				}
			}
		}
	}
	return anchors
}
