import Foundation
import CoreGraphics

final class Track {
    let id: Int
    var bbox: CGRect
    var confidence: Float
    var smoothedBBox: CGRect
    var missedFrames: Int
    var age: Int
    var confidenceAccumulator: Float

    init(id: Int, bbox: CGRect, confidence: Float) {
        self.id = id
        self.bbox = bbox
        self.confidence = confidence
        self.smoothedBBox = bbox
        self.missedFrames = 0
        self.age = 1
        self.confidenceAccumulator = confidence
    }

    func update(bbox: CGRect, confidence: Float, emaAlpha: CGFloat) {
        self.bbox = bbox
        self.confidence = confidence
        self.smoothedBBox = Track.emaRect(old: smoothedBBox, new: bbox, alpha: emaAlpha)
        self.missedFrames = 0
        self.age += 1
        self.confidenceAccumulator = 0.8 * self.confidenceAccumulator + 0.2 * confidence
    }

    static func emaRect(old: CGRect, new: CGRect, alpha: CGFloat) -> CGRect {
        let a = max(0, min(1, alpha))
        let x = old.origin.x * (1 - a) + new.origin.x * a
        let y = old.origin.y * (1 - a) + new.origin.y * a
        let w = old.width * (1 - a) + new.width * a
        let h = old.height * (1 - a) + new.height * a
        return CGRect(x: x, y: y, width: w, height: h)
    }
}
