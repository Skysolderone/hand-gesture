package detector

import (
	"image"
	"image/color"
	"math"

	"v1/cbridge"
	"gocv.io/x/gocv"
)

const LandmarkPatchSize = 224

// LandmarkResult holds detected 21 keypoints in original-frame pixel coords.
type LandmarkResult struct {
	Points     [21][2]int // pixel coords in the original frame
	Handedness float32    // >0.5 means right hand
}

// GetHandPatch extracts the rotated 224×224 hand patch and the affine matrix used.
// The affine matrix maps original-frame pixels → patch pixels.
func GetHandPatch(frame gocv.Mat, det Detection) (gocv.Mat, gocv.Mat) {
	fw := float64(frame.Cols())
	fh := float64(frame.Rows())

	// Bounding box in pixel coords
	cx := det.CX * float32(fw)
	cy := det.CY * float32(fh)
	bw := det.W * float32(fw)
	bh := det.H * float32(fh)
	boxSize := math.Max(float64(bw), float64(bh)) * 2.6 // expansion factor

	// Rotation: wrist (kp[0]) → middle MCP (kp[2])
	kp0x := det.Keypoints[0][0] * float32(fw)
	kp0y := det.Keypoints[0][1] * float32(fh)
	kp2x := det.Keypoints[2][0] * float32(fw)
	kp2y := det.Keypoints[2][1] * float32(fh)
	angleDeg := (math.Atan2(float64(kp2y-kp0y), float64(kp2x-kp0x))-math.Pi/2) * 180 / math.Pi

	scale := float64(LandmarkPatchSize) / boxSize

	// GetRotationMatrix2D takes integer center; we compute the matrix manually
	// so that the centre is in float coordinates.
	// Build a 2×3 rotation+scale matrix:
	//   [α, β, (1-α)*cx - β*cy]
	//   [-β, α, β*cx + (1-α)*cy]
	// where α = scale*cos(angle), β = scale*sin(angle)
	// Then add translation so box centre → patch centre (half, half)
	half := float64(LandmarkPatchSize) / 2.0
	angleRad := angleDeg * math.Pi / 180.0
	cosA := scale * math.Cos(angleRad)
	sinA := scale * math.Sin(angleRad)
	fcx, fcy := float64(cx), float64(cy)
	tx := (1-cosA)*fcx + sinA*fcy + half - fcx*scale
	ty := -sinA*fcx + (1-cosA)*fcy + half - fcy*scale

	// Pack into a 2×3 Mat
	rotMat := gocv.NewMatWithSize(2, 3, gocv.MatTypeCV64F)
	rotMat.SetDoubleAt(0, 0, cosA)
	rotMat.SetDoubleAt(0, 1, sinA)
	rotMat.SetDoubleAt(0, 2, tx)
	rotMat.SetDoubleAt(1, 0, -sinA)
	rotMat.SetDoubleAt(1, 1, cosA)
	rotMat.SetDoubleAt(1, 2, ty)

	patch := gocv.NewMat()
	gocv.WarpAffineWithParams(frame, &patch, rotMat,
		image.Pt(LandmarkPatchSize, LandmarkPatchSize),
		gocv.InterpolationLinear, gocv.BorderConstant, color.RGBA{})
	return patch, rotMat
}

// DetectLandmarks runs hand landmark inference on a patch and projects
// the results back to original-frame coordinates using rotMat.
func DetectLandmarks(patch gocv.Mat, rotMat gocv.Mat) LandmarkResult {
	bytes := patch.ToBytes()
	w, h := patch.Cols(), patch.Rows()

	raw, hand := cbridge.RunHandLandmark(bytes, w, h)

	// Build inverse affine matrix (2×3) for back-projection
	invMat := gocv.NewMat()
	defer invMat.Close()
	gocv.InvertAffineTransform(rotMat, &invMat)

	m00 := invMat.GetDoubleAt(0, 0)
	m01 := invMat.GetDoubleAt(0, 1)
	m02 := invMat.GetDoubleAt(0, 2)
	m10 := invMat.GetDoubleAt(1, 0)
	m11 := invMat.GetDoubleAt(1, 1)
	m12 := invMat.GetDoubleAt(1, 2)

	var res LandmarkResult
	res.Handedness = hand
	for i := 0; i < 21; i++ {
		// raw values are in [0..224] patch pixel space
		px := float64(raw[i*3+0])
		py := float64(raw[i*3+1])
		ox := m00*px + m01*py + m02
		oy := m10*px + m11*py + m12
		res.Points[i] = [2]int{int(ox), int(oy)}
	}
	return res
}
