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
            let yoloBBox = selected.bbox
            stage += " tracks=\(tracks.count) selected=\(selected.id) yolo_bbox=\(yoloBBox.debugDescription)"

            var classificationCrop: CVPixelBuffer?
            if frameCounter % classifyEveryNFrames == 0 {
                if let crop = cropper.crop(frame: frame, bboxXYXY: yoloBBox) {
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
                               trackingBBox: yoloBBox,
                               selectedBBox: yoloBBox,
                               classificationCrop: classificationCrop,
                               debugInfo: stage)
        } else {
            stage += " tracks=\(tracks.count) selected=nil"
            return FrameOutput(label: cachedResult.label,
                               confidence: cachedResult.confidence,
                               selectedTrack: nil,
                               trackingBBox: nil,
                               selectedBBox: nil,
                               classificationCrop: nil,
                               debugInfo: stage)
        }
    }

}
