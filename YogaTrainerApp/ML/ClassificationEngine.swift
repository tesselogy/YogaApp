import Foundation
import CoreVideo

@available(*, deprecated, message: "Use PoseClassifier via FrameProcessor")
final class ClassificationEngine {
    private let classifier = PoseClassifier()

    func classify(buffer: CVPixelBuffer,
                  regionOfInterest: CGRect? = nil,
                  completion: @escaping (String, Double) -> Void) {
        guard let classifier else {
            completion("model_missing", 0)
            return
        }

        let cropBuffer: CVPixelBuffer
        if let regionOfInterest,
           let cropped = Cropper().crop(frame: buffer, bboxXYXY: regionOfInterest) {
            cropBuffer = cropped
        } else {
            cropBuffer = buffer
        }

        let result = classifier.classify(cropBuffer: cropBuffer)
        completion(result.label, Double(result.confidence))
    }
}
