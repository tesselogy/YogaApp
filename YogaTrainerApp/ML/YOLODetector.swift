import CoreML
import Foundation
import CoreVideo
import CoreImage

struct Detection {
    let bbox: CGRect // pixel xyxy
    let confidence: Float
    let classIndex: Int
}

private struct InputPreparation {
    let buffer: CVPixelBuffer
    let modelWidth: Int
    let modelHeight: Int
    let frameWidth: Int
    let frameHeight: Int
    let scale: Float
    let padX: Float
    let padY: Float
}

final class YOLODetector {
    static var lastInitError: String = ""

    private var lastDebugMessage: String = ""
    private var lastCoordsDump: String = ""
    private let model: MLModel
    private let inputName: String
    private let outputNames: [String]

    private let confidenceThreshold: Float
    private let iouThreshold: Float
    private let personClassIndex: Int
    private let inputConstraint: MLImageConstraint?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    private let coordsPrimaryFormula: String = "xywh"

    init?(modelName: String = "yolov8n",
          confidenceThreshold: Float = 0.25,
          iouThreshold: Float = 0.45,
          personClassIndex: Int = 0) {
        guard let url = Bundle.main.url(forResource: modelName, withExtension: "mlmodelc") else {
            Self.lastInitError = "model_not_found:\(modelName).mlmodelc"
            return nil
        }

        func loadModel(computeUnits: MLComputeUnits) throws -> MLModel {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            return try MLModel(contentsOf: url, configuration: config)
        }

        let loadedModel: MLModel
        do {
            loadedModel = try loadModel(computeUnits: .all)
        } catch {
            do {
                loadedModel = try loadModel(computeUnits: .cpuOnly)
                Self.lastInitError = "fallback_cpu_only_used after all_failed: \(error.localizedDescription)"
            } catch {
                Self.lastInitError = "model_load_failed: \(error.localizedDescription)"
                return nil
            }
        }

        guard let imageInput = loadedModel.modelDescription.inputDescriptionsByName.first(where: { $0.value.type == .image })?.key else {
            Self.lastInitError = "image_input_not_found"
            return nil
        }

        self.model = loadedModel
        self.inputName = imageInput
        self.outputNames = Array(loadedModel.modelDescription.outputDescriptionsByName.keys)
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
        self.personClassIndex = personClassIndex
        self.inputConstraint = loadedModel.modelDescription.inputDescriptionsByName[imageInput]?.imageConstraint
    }

    func detectPersons(in frame: CVPixelBuffer) -> [Detection] {
        guard let prepared = makeInputBufferIfNeeded(from: frame) else {
            lastDebugMessage = "input_prepare_failed"
            return []
        }

        do {
            let result = try runPrediction(prepared: prepared)
            let shapeInfo = outputNames.compactMap { name -> String? in
                guard let m = result.prediction.featureValue(for: name)?.multiArrayValue else { return nil }
                return "\(name):\(m.shape.map { $0.intValue })"
            }.joined(separator: ",")

            let maxPersonConf = result.persons.map(\.confidence).max() ?? 0
            lastDebugMessage = "fmt=\(pixelFormatName(CVPixelBufferGetPixelFormatType(frame)))->\(pixelFormatName(CVPixelBufferGetPixelFormatType(prepared.buffer))) size=\(prepared.frameWidth)x\(prepared.frameHeight)->\(prepared.modelWidth)x\(prepared.modelHeight) norm(scale=\(String(format: "%.4f", prepared.scale)) padX=\(String(format: "%.1f", prepared.padX)) padY=\(String(format: "%.1f", prepared.padY))) outputs=[\(shapeInfo)] all=\(result.all.count) person=\(result.persons.count) nms=\(result.nms.count) maxPersonConf=\(String(format: "%.3f", maxPersonConf)) \(lastCoordsDump)"
            return result.nms
        } catch {
            if error.localizedDescription.contains("not in allowed set of image sizes"),
               let fallback = fallbackPredictionWithCandidateSizes(frame: frame, originalError: error) {
                return fallback
            }
            lastDebugMessage = "prediction_failed err=\(error.localizedDescription)"
            return []
        }
    }

    func debugMessage() -> String { lastDebugMessage }

    private func runPrediction(prepared: InputPreparation) throws -> (prediction: MLFeatureProvider, all: [Detection], persons: [Detection], nms: [Detection]) {
        let provider = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: prepared.buffer)])
        let prediction = try model.prediction(from: provider)

        let all = decodeYOLOOutput(prediction: prediction, prep: prepared)
        let persons = all.filter { $0.classIndex == personClassIndex }
        let nms = nonMaximumSuppression(persons, iouThreshold: iouThreshold)
        return (prediction, all, persons, nms)
    }

    private func fallbackPredictionWithCandidateSizes(frame: CVPixelBuffer,
                                                      originalError: Error) -> [Detection]? {
        let candidates = [192, 224, 256, 320, 384, 416, 448, 512, 576, 608, 640, 672, 704, 736, 768, 800, 960, 1024, 1280]
        for side in candidates {
            guard let prepared = makePreparedBuffer(from: frame,
                                                    targetWidth: side,
                                                    targetHeight: side,
                                                    targetFormat: inputConstraint?.pixelFormatType ?? CVPixelBufferGetPixelFormatType(frame),
                                                    letterbox: true) else { continue }
            do {
                let result = try runPrediction(prepared: prepared)
                let maxPersonConf = result.persons.map(\.confidence).max() ?? 0
                lastDebugMessage = "fallback_size=\(side)x\(side) all=\(result.all.count) person=\(result.persons.count) nms=\(result.nms.count) maxPersonConf=\(String(format: "%.3f", maxPersonConf)) after_err=\(originalError.localizedDescription) \(lastCoordsDump)"
                return result.nms
            } catch {
                continue
            }
        }
        return nil
    }

    private func decodeYOLOOutput(prediction: MLFeatureProvider, prep: InputPreparation) -> [Detection] {
        let outputArrays: [String: MLMultiArray] = Dictionary(uniqueKeysWithValues: outputNames.compactMap {
            guard let m = prediction.featureValue(for: $0)?.multiArrayValue else { return nil }
            return ($0.lowercased(), m)
        })

        if let coords = outputArrays.first(where: { $0.key.contains("coord") })?.value,
           let confs = outputArrays.first(where: { $0.key.contains("conf") })?.value,
           let decoded = decodeCoordinatesConfidence(coords: coords, confs: confs, prep: prep) {
            return decoded
        }

        let multiArrays: [MLMultiArray] = outputNames.compactMap { prediction.featureValue(for: $0)?.multiArrayValue }
        lastCoordsDump = "coords_dump=na"
        guard let tensor = multiArrays.max(by: { $0.count < $1.count }) else { return [] }
        let shape = tensor.shape.map { $0.intValue }
        let values = (0..<tensor.count).map { tensor[$0].floatValue }

        // [1,84,N] / [84,N]
        if (shape.count == 3 && shape[1] > 5) || (shape.count == 2 && shape[0] > 5) {
            let channels = shape.count == 3 ? shape[1] : shape[0]
            let anchors = shape.count == 3 ? shape[2] : shape[1]
            let classCount = channels - 4
            var detections: [Detection] = []

            for a in 0..<anchors {
                let cx = values[a]
                let cy = values[anchors + a]
                let w = values[2 * anchors + a]
                let h = values[3 * anchors + a]

                var bestClass = 0
                var bestScore: Float = 0
                for c in 0..<classCount {
                    let score = values[(4 + c) * anchors + a]
                    if score > bestScore {
                        bestScore = score
                        bestClass = c
                    }
                }
                guard bestScore >= confidenceThreshold else { continue }

                let normalized = max(abs(cx), abs(cy), abs(w), abs(h)) <= 2.0
                let x1m = normalized ? (cx - w / 2) * Float(prep.modelWidth) : (cx - w / 2)
                let y1m = normalized ? (cy - h / 2) * Float(prep.modelHeight) : (cy - h / 2)
                let x2m = normalized ? (cx + w / 2) * Float(prep.modelWidth) : (cx + w / 2)
                let y2m = normalized ? (cy + h / 2) * Float(prep.modelHeight) : (cy + h / 2)

                if let box = mapModelBoxToFrame(x1m, y1m, x2m, y2m, prep: prep) {
                    detections.append(Detection(bbox: box, confidence: bestScore, classIndex: bestClass))
                }
            }
            return detections
        }

        // [N,6] x1,y1,x2,y2,conf,class
        if shape.last == 6 {
            let rows = tensor.count / 6
            var detections: [Detection] = []
            for r in 0..<rows {
                let base = r * 6
                let x1raw = values[base]
                let y1raw = values[base + 1]
                let x2raw = values[base + 2]
                let y2raw = values[base + 3]
                let conf = values[base + 4]
                let cls = Int(values[base + 5])
                guard conf >= confidenceThreshold else { continue }

                let normalized = max(abs(x1raw), abs(y1raw), abs(x2raw), abs(y2raw)) <= 2.0
                let x1m = normalized ? x1raw * Float(prep.modelWidth) : x1raw
                let y1m = normalized ? y1raw * Float(prep.modelHeight) : y1raw
                let x2m = normalized ? x2raw * Float(prep.modelWidth) : x2raw
                let y2m = normalized ? y2raw * Float(prep.modelHeight) : y2raw

                if let box = mapModelBoxToFrame(x1m, y1m, x2m, y2m, prep: prep) {
                    detections.append(Detection(bbox: box, confidence: conf, classIndex: cls))
                }
            }
            return detections
        }

        lastDebugMessage = "unsupported_tensor_shape=\(shape)"
        return []
    }

    private func decodeCoordinatesConfidence(coords: MLMultiArray,
                                             confs: MLMultiArray,
                                             prep: InputPreparation) -> [Detection]? {
        let coordShape = coords.shape.map { $0.intValue }
        let confShape = confs.shape.map { $0.intValue }
        guard coordShape.last == 4, let classes = confShape.last, classes > 0 else { return nil }

        let rows = min(coords.count / 4, confs.count / classes)
        guard rows > 0 else { return nil }

        let coordVals = (0..<coords.count).map { coords[$0].floatValue }
        let confVals = (0..<confs.count).map { confs[$0].floatValue }

        var detections: [Detection] = []
        var dumps: [String] = []

        for r in 0..<rows {
            let cbase = r * 4
            let a = coordVals[cbase]
            let b = coordVals[cbase + 1]
            let c = coordVals[cbase + 2]
            let d = coordVals[cbase + 3]

            var bestClass = 0
            var bestScore: Float = -Float.greatestFiniteMagnitude
            let pbase = r * classes
            for cls in 0..<classes {
                let score = confVals[pbase + cls]
                if score > bestScore {
                    bestScore = score
                    bestClass = cls
                }
            }

            let normalized = max(abs(a), abs(b), abs(c), abs(d)) <= 2.0

            // Interpret as XYXY
            let x1xyxy = normalized ? a * Float(prep.modelWidth) : a
            let y1xyxy = normalized ? b * Float(prep.modelHeight) : b
            let x2xyxy = normalized ? c * Float(prep.modelWidth) : c
            let y2xyxy = normalized ? d * Float(prep.modelHeight) : d
            let mappedXYXY = mapModelBoxToFrame(x1xyxy, y1xyxy, x2xyxy, y2xyxy, prep: prep)

            // Interpret as XYWH
            let cx = normalized ? a * Float(prep.modelWidth) : a
            let cy = normalized ? b * Float(prep.modelHeight) : b
            let w = normalized ? c * Float(prep.modelWidth) : c
            let h = normalized ? d * Float(prep.modelHeight) : d
            let x1xywh = cx - w / 2
            let y1xywh = cy - h / 2
            let x2xywh = cx + w / 2
            let y2xywh = cy + h / 2
            let mappedXYWH = mapModelBoxToFrame(x1xywh, y1xywh, x2xywh, y2xywh, prep: prep)

            let chosen: CGRect?
            if coordsPrimaryFormula == "xywh" {
                chosen = mappedXYWH ?? mappedXYXY
            } else {
                chosen = mappedXYXY ?? mappedXYWH
            }

            let dump = "r\(r):raw=[\(String(format: "%.2f", a)),\(String(format: "%.2f", b)),\(String(format: "%.2f", c)),\(String(format: "%.2f", d))] norm=\(normalized ? 1 : 0) top=\(bestClass):\(String(format: "%.3f", bestScore)) xyxy=\(mappedXYXY?.debugDescription ?? "nil") xywh=\(mappedXYWH?.debugDescription ?? "nil") chosen=\(chosen?.debugDescription ?? "nil") formula=\(coordsPrimaryFormula)"
            dumps.append(dump)

            guard bestScore >= confidenceThreshold else { continue }
            if let box = chosen {
                detections.append(Detection(bbox: box, confidence: bestScore, classIndex: bestClass))
            }
        }

        let space = "space(camera=\(prep.frameWidth)x\(prep.frameHeight),model=\(prep.modelWidth)x\(prep.modelHeight),scale=\(String(format: "%.4f", prep.scale)),pad=[\(String(format: "%.1f", prep.padX)),\(String(format: "%.1f", prep.padY))])"
        lastCoordsDump = "coords_formula_probe formula=\(coordsPrimaryFormula) \(space) rows=\(rows) \(dumps.joined(separator: " | "))"

        return detections
    }

    private func mapModelBoxToFrame(_ x1m: Float, _ y1m: Float, _ x2m: Float, _ y2m: Float, prep: InputPreparation) -> CGRect? {
        let x1f = (x1m - prep.padX) / prep.scale
        let y1f = (y1m - prep.padY) / prep.scale
        let x2f = (x2m - prep.padX) / prep.scale
        let y2f = (y2m - prep.padY) / prep.scale

        let x1 = max(0, min(Float(prep.frameWidth - 1), x1f))
        let y1 = max(0, min(Float(prep.frameHeight - 1), y1f))
        let x2 = max(0, min(Float(prep.frameWidth - 1), x2f))
        let y2 = max(0, min(Float(prep.frameHeight - 1), y2f))
        guard x2 > x1, y2 > y1 else { return nil }

        return CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(x2 - x1), height: CGFloat(y2 - y1))
    }

    private func makeInputBufferIfNeeded(from frame: CVPixelBuffer) -> InputPreparation? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)

        guard let inputConstraint else {
            return InputPreparation(buffer: frame,
                                    modelWidth: frameWidth,
                                    modelHeight: frameHeight,
                                    frameWidth: frameWidth,
                                    frameHeight: frameHeight,
                                    scale: 1,
                                    padX: 0,
                                    padY: 0)
        }

        let expectedFormat = inputConstraint.pixelFormatType
        let actualFormat = CVPixelBufferGetPixelFormatType(frame)
        let modelW = inputConstraint.pixelsWide > 0 ? inputConstraint.pixelsWide : frameWidth
        let modelH = inputConstraint.pixelsHigh > 0 ? inputConstraint.pixelsHigh : frameHeight

        return makePreparedBuffer(from: frame,
                                  targetWidth: modelW,
                                  targetHeight: modelH,
                                  targetFormat: expectedFormat == 0 ? actualFormat : expectedFormat,
                                  letterbox: true)
    }

    private func makePreparedBuffer(from frame: CVPixelBuffer,
                                    targetWidth: Int,
                                    targetHeight: Int,
                                    targetFormat: OSType,
                                    letterbox: Bool) -> InputPreparation? {
        let frameWidth = CVPixelBufferGetWidth(frame)
        let frameHeight = CVPixelBufferGetHeight(frame)
        let inFormat = CVPixelBufferGetPixelFormatType(frame)
        let outFormat = targetFormat == 0 ? inFormat : targetFormat

        if frameWidth == targetWidth, frameHeight == targetHeight, inFormat == outFormat {
            return InputPreparation(buffer: frame,
                                    modelWidth: targetWidth,
                                    modelHeight: targetHeight,
                                    frameWidth: frameWidth,
                                    frameHeight: frameHeight,
                                    scale: 1,
                                    padX: 0,
                                    padY: 0)
        }

        var converted: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true
        ]

        guard CVPixelBufferCreate(kCFAllocatorDefault,
                                  targetWidth,
                                  targetHeight,
                                  outFormat,
                                  attrs as CFDictionary,
                                  &converted) == kCVReturnSuccess,
              let converted else {
            return nil
        }

        let ciImage = CIImage(cvPixelBuffer: frame)

        if letterbox {
            let scale = min(CGFloat(targetWidth) / CGFloat(frameWidth), CGFloat(targetHeight) / CGFloat(frameHeight))
            let resizedW = CGFloat(frameWidth) * scale
            let resizedH = CGFloat(frameHeight) * scale
            let padX = (CGFloat(targetWidth) - resizedW) / 2
            let padY = (CGFloat(targetHeight) - resizedH) / 2

            // gray fill like Ultralytics letterbox
            let bg = CIImage(color: CIColor(red: 0.447, green: 0.447, blue: 0.447, alpha: 1))
                .cropped(to: CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight))

            let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
                .transformed(by: CGAffineTransform(translationX: padX, y: padY))
            let composed = resized.composited(over: bg)
            ciContext.render(composed, to: converted)

            return InputPreparation(buffer: converted,
                                    modelWidth: targetWidth,
                                    modelHeight: targetHeight,
                                    frameWidth: frameWidth,
                                    frameHeight: frameHeight,
                                    scale: Float(scale),
                                    padX: Float(padX),
                                    padY: Float(padY))
        } else {
            let scaleX = CGFloat(targetWidth) / CGFloat(frameWidth)
            let scaleY = CGFloat(targetHeight) / CGFloat(frameHeight)
            let resized = ciImage.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
            ciContext.render(resized, to: converted)

            return InputPreparation(buffer: converted,
                                    modelWidth: targetWidth,
                                    modelHeight: targetHeight,
                                    frameWidth: frameWidth,
                                    frameHeight: frameHeight,
                                    scale: Float(scaleX),
                                    padX: 0,
                                    padY: 0)
        }
    }

    private func pixelFormatName(_ type: OSType) -> String {
        switch type {
        case kCVPixelFormatType_32BGRA: return "32BGRA"
        case kCVPixelFormatType_OneComponent8: return "OneComponent8"
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange: return "420f"
        case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange: return "420v"
        case 0: return "unspecified"
        default: return "\(type)"
        }
    }

    private func nonMaximumSuppression(_ detections: [Detection], iouThreshold: Float) -> [Detection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var selected: [Detection] = []

        for det in sorted {
            var keep = true
            for chosen in selected {
                if iou(det.bbox, chosen.bbox) > iouThreshold {
                    keep = false
                    break
                }
            }
            if keep { selected.append(det) }
        }

        return selected
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = Float(inter.width * inter.height)
        let unionArea = Float(a.width * a.height + b.width * b.height - inter.width * inter.height)
        return unionArea > 0 ? interArea / unionArea : 0
    }
}
