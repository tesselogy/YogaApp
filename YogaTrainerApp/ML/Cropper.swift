import Foundation
import CoreVideo

final class Cropper {
    func crop(frame: CVPixelBuffer, bboxXYXY: CGRect) -> CVPixelBuffer? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)

        let x1 = max(0, min(frameWidth - 1, Int(bboxXYXY.minX)))
        let y1 = max(0, min(frameHeight - 1, Int(bboxXYXY.minY)))
        let x2 = max(0, min(frameWidth, Int(bboxXYXY.maxX)))
        let y2 = max(0, min(frameHeight, Int(bboxXYXY.maxY)))

        let cropWidth = x2 - x1
        let cropHeight = y2 - y1
        guard cropWidth > 0, cropHeight > 0 else { return nil }

        let format = CVPixelBufferGetPixelFormatType(frame)
        guard format == kCVPixelFormatType_32BGRA else { return nil }

        var outBuffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        guard CVPixelBufferCreate(kCFAllocatorDefault,
                                  cropWidth,
                                  cropHeight,
                                  format,
                                  attrs as CFDictionary,
                                  &outBuffer) == kCVReturnSuccess,
              let cropped = outBuffer else {
            return nil
        }

        CVPixelBufferLockBaseAddress(frame, .readOnly)
        CVPixelBufferLockBaseAddress(cropped, [])
        defer {
            CVPixelBufferUnlockBaseAddress(cropped, [])
            CVPixelBufferUnlockBaseAddress(frame, .readOnly)
        }

        guard let srcBase = CVPixelBufferGetBaseAddress(frame),
              let dstBase = CVPixelBufferGetBaseAddress(cropped) else {
            return nil
        }

        let srcStride = CVPixelBufferGetBytesPerRow(frame)
        let dstStride = CVPixelBufferGetBytesPerRow(cropped)
        let bytesPerPixel = 4

        for row in 0..<cropHeight {
            let srcRow = srcBase.advanced(by: (y1 + row) * srcStride + x1 * bytesPerPixel)
            let dstRow = dstBase.advanced(by: row * dstStride)
            memcpy(dstRow, srcRow, cropWidth * bytesPerPixel)
        }

        return cropped
    }
}
