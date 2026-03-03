import Foundation
import CoreVideo

@available(*, deprecated, message: "Use YOLODetector + ByteTracker via FrameProcessor")
final class FocusTrackingEngine {
    private let detector = YOLODetector()

    func process(buffer: CVPixelBuffer,
                 completion: @escaping (CGRect?) -> Void) {
        guard let detector else {
            completion(nil)
            return
        }

        let det = detector.detectPersons(in: buffer).max(by: { $0.confidence < $1.confidence })
        completion(det?.bbox)
    }
}
