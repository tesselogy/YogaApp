import Foundation
import CoreVideo

struct FrameOutput {
    let label: String
    let confidence: Float
    let selectedTrack: Track?
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

        if let selected,
           frameCounter % classifyEveryNFrames == 0,
           let crop = cropper.crop(frame: frame, bboxXYXY: selected.smoothedBBox) {
            cachedResult = classifier.classify(cropBuffer: crop)
        }

        return FrameOutput(label: cachedResult.label, confidence: cachedResult.confidence, selectedTrack: selected)
    }
}
