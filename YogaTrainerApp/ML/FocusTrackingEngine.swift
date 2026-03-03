import Vision
import CoreML
import Foundation

struct YTDetectedPerson {
    let id: Int
    let observation: VNDetectedObjectObservation
    let confidence: Double
}

struct YTFocusResult {
    let people: [YTDetectedPerson]
    let active: YTDetectedPerson?
}

class YTFocusTrackingEngine {

    private var detectionModel: VNCoreMLModel?
    private var lastLogTime: Date = .distantPast

    private var activeTrackID: Int?
    private var focusLockUntil: Date?
    private var nextID: Int = 1
    private var previousPeople: [YTDetectedPerson] = []
    private var emaBox: CGRect?

    init() {
        detectionModel = Self.loadModel(named: "yolov8n")
    }

    func process(buffer: CVPixelBuffer,
                 completion: @escaping (YTFocusResult) -> Void) {
        detect(buffer: buffer) { people in
            let withIDs = self.assignPersistentIDs(to: people)
            let active = self.selectActiveSubject(from: withIDs)

            completion(YTFocusResult(people: withIDs, active: active))
        }
    }

    private func detect(buffer: CVPixelBuffer,
                        completion: @escaping ([YTDetectedPerson]) -> Void) {

        guard let model = detectionModel else {
            detectHumanFallback(buffer: buffer, completion: completion)
            return
        }

        let request = VNCoreMLRequest(model: model) { request, _ in
            let results = request.results as? [VNRecognizedObjectObservation] ?? []

            if results.isEmpty {
                self.detectHumanFallback(buffer: buffer, completion: completion)
                return
            }

            let persons = results.filter { $0.labels.first?.identifier.lowercased() == "person" }
            let candidates = persons.isEmpty ? results : persons

            let detected = candidates.map { obs in
                let expanded = self.expandForFullBody(obs.boundingBox)
                return YTDetectedPerson(id: -1,
                                      observation: VNDetectedObjectObservation(boundingBox: expanded),
                                      confidence: Double(obs.confidence))
            }

            completion(detected)
        }

        request.imageCropAndScaleOption = .scaleFill

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Detection request failed: \(error.localizedDescription)")
            detectHumanFallback(buffer: buffer, completion: completion)
        }
    }

    private func detectHumanFallback(buffer: CVPixelBuffer,
                                     completion: @escaping ([YTDetectedPerson]) -> Void) {
        let request = VNDetectHumanRectanglesRequest { request, _ in
            let humans = request.results as? [VNHumanObservation] ?? []
            let detected = humans.map {
                let expanded = self.expandForFullBody($0.boundingBox)
                return YTDetectedPerson(id: -1,
                                      observation: VNDetectedObjectObservation(boundingBox: expanded),
                                      confidence: 0.55)
            }
            completion(detected)
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: buffer)
        do {
            try handler.perform([request])
        } catch {
            debugLog("Vision fallback failed: \(error.localizedDescription)")
            completion([])
        }
    }

    private func assignPersistentIDs(to people: [YTDetectedPerson]) -> [YTDetectedPerson] {
        var assigned: [YTDetectedPerson] = []

        for person in people {
            let bbox = person.observation.boundingBox
            if let matched = previousPeople.max(by: {
                iou($0.observation.boundingBox, bbox) < iou($1.observation.boundingBox, bbox)
            }), iou(matched.observation.boundingBox, bbox) > 0.2 {
                assigned.append(YTDetectedPerson(id: matched.id,
                                               observation: person.observation,
                                               confidence: person.confidence))
            } else {
                assigned.append(YTDetectedPerson(id: nextID,
                                               observation: person.observation,
                                               confidence: person.confidence))
                nextID += 1
            }
        }

        previousPeople = assigned
        return assigned
    }

    private func selectActiveSubject(from people: [YTDetectedPerson]) -> YTDetectedPerson? {
        guard !people.isEmpty else {
            activeTrackID = nil
            focusLockUntil = nil
            emaBox = nil
            return nil
        }

        let now = Date()

        if let currentID = activeTrackID,
           let locked = people.first(where: { $0.id == currentID }),
           let lockUntil = focusLockUntil,
           now < lockUntil {
            return smoothed(person: locked)
        }

        let scored = people.max { lhs, rhs in
            subjectScore(lhs.observation.boundingBox) < subjectScore(rhs.observation.boundingBox)
        }

        if let scored, scored.id != activeTrackID {
            activeTrackID = scored.id
            focusLockUntil = now.addingTimeInterval(2.0)
            emaBox = scored.observation.boundingBox
        }

        return scored.map(smoothed(person:))
    }

    private func smoothed(person: YTDetectedPerson) -> YTDetectedPerson {
        let current = person.observation.boundingBox
        let alpha: CGFloat = 0.25

        let smoothedBox: CGRect
        if let previous = emaBox {
            smoothedBox = CGRect(
                x: previous.origin.x * (1 - alpha) + current.origin.x * alpha,
                y: previous.origin.y * (1 - alpha) + current.origin.y * alpha,
                width: previous.size.width * (1 - alpha) + current.size.width * alpha,
                height: previous.size.height * (1 - alpha) + current.size.height * alpha
            )
        } else {
            smoothedBox = current
        }

        emaBox = smoothedBox
        return YTDetectedPerson(id: person.id,
                              observation: VNDetectedObjectObservation(boundingBox: smoothedBox),
                              confidence: person.confidence)
    }

    private func subjectScore(_ bbox: CGRect) -> CGFloat {
        let center = CGPoint(x: bbox.midX, y: bbox.midY)
        let dx = center.x - 0.5
        let dy = center.y - 0.5
        let distance = sqrt(dx * dx + dy * dy)
        let centerScore = max(0, 1 - distance / 0.8)
        let areaScore = bbox.width * bbox.height
        return centerScore * 0.6 + areaScore * 0.4
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        return unionArea > 0 ? interArea / unionArea : 0
    }

    private func expandForFullBody(_ bbox: CGRect) -> CGRect {
        let widthScale: CGFloat = 1.35
        let heightScale: CGFloat = 2.0

        let expandedWidth = min(1, bbox.width * widthScale)
        let expandedHeight = min(1, bbox.height * heightScale)

        let centerX = bbox.midX
        let centerY = bbox.midY - bbox.height * 0.15

        let x = max(0, min(1 - expandedWidth, centerX - expandedWidth / 2))
        let y = max(0, min(1 - expandedHeight, centerY - expandedHeight / 2))

        return CGRect(x: x, y: y, width: expandedWidth, height: expandedHeight)
    }

    private static func loadModel(named name: String) -> VNCoreMLModel? {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc"),
              let model = try? MLModel(contentsOf: url) else {
            return nil
        }

        return try? VNCoreMLModel(for: model)
    }

    private func debugLog(_ message: String) {
        let now = Date()
        guard now.timeIntervalSince(lastLogTime) > 1 else { return }
        lastLogTime = now
        print("[FocusTrackingEngine] \(message)")
    }
}
