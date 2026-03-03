import Foundation
import CoreVideo

struct FrameOutput {
    let label: String
    let confidence: Float
    let selectedTrack: Track?
    let debugInfo: String
}

final class FrameProcessor {
    private let detector: YOLODetector
    private let tracker: ByteTracker
    private let selector: CandidateSelector
    private let cropper: Cropper
    private let classifier: PoseClassifier

    private let classifyEveryNFrames: Int
    private var frameCounter: Int = 0
    private var cachedResult: ClassificationResult = .init(label: "...", confidence: 0)

    init?(classifyEveryNFrames: Int = 3) {
        guard let detector = YOLODetector(),
              let classifier = PoseClassifier() else {
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
            stage += " tracks=\(tracks.count) selected=\(selected.id) bbox=\(selected.smoothedBBox.debugDescription)"
            if frameCounter % classifyEveryNFrames == 0 {
                if let crop = cropper.crop(frame: frame, bboxXYXY: selected.smoothedBBox) {
                    cachedResult = classifier.classify(cropBuffer: crop)
                    stage += " crop=ok classify=\(cachedResult.label):\(String(format: "%.2f", cachedResult.confidence))"
                } else {
                    stage += " crop=empty"
                }
            } else {
                stage += " classify=skip(\(frameCounter)%\(classifyEveryNFrames))"
            }
        } else {
            stage += " tracks=\(tracks.count) selected=nil"
        }

        return FrameOutput(label: cachedResult.label, confidence: cachedResult.confidence, selectedTrack: selected, debugInfo: stage)
    }
}
