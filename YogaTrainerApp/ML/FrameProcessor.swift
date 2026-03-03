import Foundation
import CoreVideo
import CoreGraphics

struct FrameOutput {
    let label: String
    let confidence: Float
    let selectedTrack: Track?
    let trackingBBox: CGRect?
    let selectedBBox: CGRect?
    let classificationCrop: CVPixelBuffer?
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
    private var lastExpandedBBox: CGRect?

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
            let rawExpanded = expandedPersonBox(selected.smoothedBBox, frameWidth: width, frameHeight: height)
            let expanded = stabilizeExpandedBox(rawExpanded, frameWidth: width, frameHeight: height)
            stage += " tracks=\(tracks.count) selected=\(selected.id) bbox=\(selected.smoothedBBox.debugDescription) expanded=\(expanded.debugDescription)"

            var classificationCrop: CVPixelBuffer?
            if frameCounter % classifyEveryNFrames == 0 {
                if let crop = cropper.crop(frame: frame, bboxXYXY: expanded) {
                    cachedResult = classifier.classify(cropBuffer: crop)
                    classificationCrop = crop
                    stage += " crop=ok classify=\(cachedResult.label):\(String(format: "%.2f", cachedResult.confidence)) clsdbg=\(cachedResult.debug)"
                } else {
                    stage += " crop=empty"
                }
            } else {
                stage += " classify=skip(\(frameCounter)%\(classifyEveryNFrames))"
            }
            return FrameOutput(label: cachedResult.label,
                               confidence: cachedResult.confidence,
                               selectedTrack: selected,
                               trackingBBox: selected.smoothedBBox,
                               selectedBBox: expanded,
                               classificationCrop: classificationCrop,
                               debugInfo: stage)
        } else {
            stage += " tracks=\(tracks.count) selected=nil"
            lastExpandedBBox = nil
            return FrameOutput(label: cachedResult.label,
                               confidence: cachedResult.confidence,
                               selectedTrack: nil,
                               trackingBBox: nil,
                               selectedBBox: nil,
                               classificationCrop: nil,
                               debugInfo: stage)
        }
    }

    private func stabilizeExpandedBox(_ box: CGRect, frameWidth: Int, frameHeight: Int) -> CGRect {
        guard let previous = lastExpandedBBox else {
            lastExpandedBBox = box
            return box
        }

        let shrinkLimit: CGFloat = 0.92
        let growLimit: CGFloat = 1.12

        func clampScale(new: CGFloat, old: CGFloat) -> CGFloat {
            guard old > 0 else { return new }
            let ratio = new / old
            if ratio < shrinkLimit { return old * shrinkLimit }
            if ratio > growLimit { return old * growLimit }
            return new
        }

        let w = clampScale(new: box.width, old: previous.width)
        let h = clampScale(new: box.height, old: previous.height)

        let cx = previous.midX * 0.35 + box.midX * 0.65
        let cy = previous.midY * 0.35 + box.midY * 0.65

        let x = max(0, min(CGFloat(frameWidth) - w, cx - w / 2))
        let y = max(0, min(CGFloat(frameHeight) - h, cy - h / 2))

        let stabilized = CGRect(x: x, y: y, width: w, height: h)
        lastExpandedBBox = stabilized
        return stabilized
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
