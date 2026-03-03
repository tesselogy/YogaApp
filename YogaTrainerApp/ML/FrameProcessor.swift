import Foundation
import CoreVideo
import CoreGraphics

struct FrameOutput {
    let label: String
    let confidence: Float
    let selectedTrack: Track?
    let selectedBBox: CGRect?
    let debugInfo: String
}

final class FrameProcessor {
    static var lastInitError: String = ""
    private let detector: YOLODetector
    private let tracker: ByteTracker
    private let selector: CandidateSelector
    private let cropper: Cropper
    private let classifier: PoseClassifier

    private let classifyEveryNFrames: Int
    private var frameCounter: Int = 0
    private var cachedResult: ClassificationResult = .init(label: "...", confidence: 0, debug: "cold_start")

    init?(classifyEveryNFrames: Int = 3) {
        guard let detector = YOLODetector() else {
            Self.lastInitError = "detector_init_failed: \(YOLODetector.lastInitError)"
            return nil
        }

        guard let classifier = PoseClassifier() else {
            Self.lastInitError = "classifier_init_failed: \(PoseClassifier.lastInitError)"
            return nil
        }

        self.detector = detector
        self.classifier = classifier
        self.tracker = ByteTracker()
        self.selector = CandidateSelector()
        self.cropper = Cropper()
        self.classifyEveryNFrames = max(1, classifyEveryNFrames)
    }

    func process(frame: CVPixelBuffer) -> FrameOutput {
        frameCounter += 1

        let detections = detector.detectPersons(in: frame)
        let tracks = tracker.update(with: detections)

        let width = CVPixelBufferGetWidth(frame)
        let height = CVPixelBufferGetHeight(frame)
        let selected = selector.selectBest(from: tracks, frameWidth: width, frameHeight: height)

        var stage = detector.debugMessage()

        if let selected {
            let expanded = expandedPersonBox(selected.smoothedBBox, frameWidth: width, frameHeight: height)
            stage += " tracks=\(tracks.count) selected=\(selected.id) bbox=\(selected.smoothedBBox.debugDescription) expanded=\(expanded.debugDescription)"
            if frameCounter % classifyEveryNFrames == 0 {
                if let crop = cropper.crop(frame: frame, bboxXYXY: expanded) {
                    cachedResult = classifier.classify(cropBuffer: crop)
                    stage += " crop=ok classify=\(cachedResult.label):\(String(format: "%.2f", cachedResult.confidence)) clsdbg=\(cachedResult.debug)"
                } else {
                    stage += " crop=empty"
                }
            } else {
                stage += " classify=skip(\(frameCounter)%\(classifyEveryNFrames))"
            }
            return FrameOutput(label: cachedResult.label, confidence: cachedResult.confidence, selectedTrack: selected, selectedBBox: expanded, debugInfo: stage)
        } else {
            stage += " tracks=\(tracks.count) selected=nil"
            return FrameOutput(label: cachedResult.label, confidence: cachedResult.confidence, selectedTrack: nil, selectedBBox: nil, debugInfo: stage)
        }
    }

    private func expandedPersonBox(_ box: CGRect, frameWidth: Int, frameHeight: Int) -> CGRect {
        let widthScale: CGFloat = 1.3
        let heightScale: CGFloat = 1.9

        let newW = min(CGFloat(frameWidth), box.width * widthScale)
        let newH = min(CGFloat(frameHeight), box.height * heightScale)

        let cx = box.midX
        let cy = box.midY - box.height * 0.12

        let x = max(0, min(CGFloat(frameWidth) - newW, cx - newW / 2))
        let y = max(0, min(CGFloat(frameHeight) - newH, cy - newH / 2))
        return CGRect(x: x, y: y, width: newW, height: newH)
    }
}
