import SwiftUI
import Vision
import Foundation

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    let focusEngine = FocusTrackingEngine()
    let classifier = ClassificationEngine()

    @State var trackedObservation: VNDetectedObjectObservation?
    @State var classificationROI: CGRect?
    @State var detectionStatus: String = "Waiting for person..."
    @State var isProcessingFrame: Bool = false
    @State private var frameCounter: Int = 0
    @State private var imageSize: CGSize = .zero

    // Pipeline stabilizers
    @State private var lastInferenceAt: Date = .distantPast
    @State private var lockedObservation: VNDetectedObjectObservation?
    @State private var focusLockUntil: Date = .distantPast
    @State private var smoothedROI: CGRect?
    @State private var voteWindow: [String] = []
    @State private var pendingPose: String?
    @State private var pendingPoseSince: Date = .distantPast

    private let processingQueue = DispatchQueue(label: "vision.processing.queue", qos: .userInitiated)

    private let targetInferenceFPS: Double = 10
    private let confidenceThreshold: Double = 0.6
    private let majorityWindowSize: Int = 7
    private let majorityMinVotes: Int = 4
    private let focusLockDuration: TimeInterval = 2.0
    private let poseSwitchHysteresis: TimeInterval = 0.45
    private let roiInterpolationAlpha: CGFloat = 0.35

    private func expandedROI(from bbox: CGRect) -> CGRect {
        let widthScale: CGFloat = 2.0
        let heightScale: CGFloat = 2.3
        let newWidth = min(1, bbox.width * widthScale)
        let newHeight = min(1, bbox.height * heightScale)
        let newX = max(0, min(1 - newWidth, bbox.midX - newWidth / 2))
        let newY = max(0, min(1 - newHeight, bbox.midY - newHeight / 2))
        return CGRect(x: newX, y: newY, width: newWidth, height: newHeight)
    }

    var body: some View {

        GeometryReader { geo in
            ZStack {

                CameraPreview(session: camera.captureSession)
                    .scaledToFill()
                    .onChange(of: camera.currentBuffer) { _, newBuffer in
                        guard let buffer = newBuffer else { return }
                        guard !isProcessingFrame else { return }

                        let now = Date()
                        let minInterval = 1.0 / targetInferenceFPS
                        guard now.timeIntervalSince(lastInferenceAt) >= minInterval else { return }
                        lastInferenceAt = now

                        imageSize = CGSize(width: CVPixelBufferGetWidth(buffer),
                                           height: CVPixelBufferGetHeight(buffer))

                        isProcessingFrame = true
                        frameCounter += 1

                        let lastLockedObservation = lockedObservation
                        let currentlyLockedObservation = (now < focusLockUntil) ? lastLockedObservation : nil
                        let shouldRedetect = currentlyLockedObservation == nil && (lastLockedObservation == nil || frameCounter % 3 == 0)

                        processingQueue.async {
                            if shouldRedetect {
                                focusEngine.process(buffer: buffer) { obs in
                                    guard let obs else {
                                        DispatchQueue.main.async {
                                            if let lockObs = lockedObservation, Date() < focusLockUntil {
                                                classify(buffer: buffer, observation: lockObs)
                                            } else {
                                                trackedObservation = nil
                                                classificationROI = nil
                                                smoothedROI = nil
                                                detectionStatus = "Person not detected"
                                                pushVote("unknown")
                                                applyPoseFromVotes()
                                                isProcessingFrame = false
                                            }
                                        }
                                        return
                                    }

                                    DispatchQueue.main.async {
                                        lockedObservation = obs
                                        focusLockUntil = Date().addingTimeInterval(focusLockDuration)
                                        classify(buffer: buffer, observation: obs)
                                    }
                                }
                            } else if let lockObs = currentlyLockedObservation {
                                DispatchQueue.main.async {
                                    classify(buffer: buffer, observation: lockObs)
                                }
                            } else if let lastObs = lastLockedObservation {
                                DispatchQueue.main.async {
                                    classify(buffer: buffer, observation: lastObs)
                                }
                            } else {
                                DispatchQueue.main.async {
                                    isProcessingFrame = false
                                }
                            }
                        }
                    }

                if let trackedObservation {
                    let rect = toPreviewRect(normalizedBBox: trackedObservation.boundingBox, viewSize: geo.size)
                    DetectionBox(observation: trackedObservation)
                        .stroke(.green, lineWidth: 4)
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }

                if let classificationROI {
                    let rect = toPreviewRect(normalizedBBox: classificationROI, viewSize: geo.size)
                    DetectionBox(observation: VNDetectedObjectObservation(boundingBox: classificationROI))
                        .stroke(.yellow, style: StrokeStyle(lineWidth: 3, dash: [10, 8]))
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }

                VStack {
                    HStack {
                        Spacer()

                        if classificationROI != nil {
                            Text("ROI locked + smoothed + voted")
                                .font(.caption)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(.black.opacity(0.45))
                                .foregroundColor(.yellow)
                                .cornerRadius(10)
                                .padding(.top, 30)
                                .padding(.trailing, 20)
                        }
                    }

                    Spacer()

                    Text(poseState.pose.uppercased())
                        .font(.system(size: 80, weight: .bold))
                        .foregroundStyle(
                            LinearGradient(
                                colors: [.purple, .cyan, .mint],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .shadow(radius: 15)

                    Text("Hold: \(poseState.holdTime, specifier: "%.1f")s")
                        .font(.title3)
                        .foregroundColor(.white.opacity(0.8))

                    Text(detectionStatus)
                        .font(.headline)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(.black.opacity(0.45))
                        .foregroundColor(.white)
                        .cornerRadius(12)

                    Spacer()

                    Text("Press Q to quit")
                        .font(.caption)
                        .padding(12)
                        .background(.ultraThinMaterial)
                        .cornerRadius(18)
                        .padding(.bottom, 40)
                }
            }
            .ignoresSafeArea()
        }
    }

    private func classify(buffer: CVPixelBuffer, observation: VNDetectedObjectObservation) {
        let rawROI = expandedROI(from: observation.boundingBox)

        classifier.classify(buffer: buffer, regionOfInterest: rawROI) { label, confidence in
            DispatchQueue.main.async {
                let stabilized = stabilizeROI(rawROI)

                trackedObservation = VNDetectedObjectObservation(boundingBox: stabilized)
                classificationROI = stabilized
                detectionStatus = "Person detected"

                if confidence >= confidenceThreshold {
                    pushVote(label)
                } else {
                    detectionStatus = "Low confidence (\(String(format: "%.2f", confidence)))"
                    pushVote("unknown")
                }

                applyPoseFromVotes()

                print("[ContentView] pose=\(label) confidence=\(String(format: "%.2f", confidence)) status=\(detectionStatus)")
                isProcessingFrame = false
            }
        }
    }

    private func stabilizeROI(_ target: CGRect) -> CGRect {
        guard let previous = smoothedROI else {
            smoothedROI = target
            return target
        }

        let blended = CGRect(
            x: previous.origin.x + (target.origin.x - previous.origin.x) * roiInterpolationAlpha,
            y: previous.origin.y + (target.origin.y - previous.origin.y) * roiInterpolationAlpha,
            width: previous.width + (target.width - previous.width) * roiInterpolationAlpha,
            height: previous.height + (target.height - previous.height) * roiInterpolationAlpha
        )

        let clamped = CGRect(
            x: max(0, min(1 - blended.width, blended.origin.x)),
            y: max(0, min(1 - blended.height, blended.origin.y)),
            width: min(1, max(0.05, blended.width)),
            height: min(1, max(0.05, blended.height))
        )

        smoothedROI = clamped
        return clamped
    }

    private func pushVote(_ label: String) {
        voteWindow.append(label)
        if voteWindow.count > majorityWindowSize {
            voteWindow.removeFirst(voteWindow.count - majorityWindowSize)
        }
    }

    private func applyPoseFromVotes() {
        let validVotes = voteWindow.filter { $0 != "unknown" && $0 != "no_person" && $0 != "model_missing" }

        guard !validVotes.isEmpty else {
            if poseState.pose != "no_person" {
                poseState.update(newPose: "no_person")
            }
            pendingPose = nil
            return
        }

        let grouped = Dictionary(grouping: validVotes, by: { $0 })
        guard let winner = grouped.max(by: { $0.value.count < $1.value.count }) else { return }
        guard winner.value.count >= majorityMinVotes else { return }

        let targetPose = winner.key
        let now = Date()

        if poseState.pose == targetPose {
            pendingPose = nil
            return
        }

        if pendingPose != targetPose {
            pendingPose = targetPose
            pendingPoseSince = now
            return
        }

        guard now.timeIntervalSince(pendingPoseSince) >= poseSwitchHysteresis else { return }

        poseState.update(newPose: targetPose)
        pendingPose = nil
    }

    private func toPreviewRect(normalizedBBox: CGRect, viewSize: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0, viewSize.width > 0, viewSize.height > 0 else {
            return CGRect(x: normalizedBBox.minX * viewSize.width,
                          y: (1 - normalizedBBox.maxY) * viewSize.height,
                          width: normalizedBBox.width * viewSize.width,
                          height: normalizedBBox.height * viewSize.height)
        }

        let scale = max(viewSize.width / imageSize.width, viewSize.height / imageSize.height)
        let scaledWidth = imageSize.width * scale
        let scaledHeight = imageSize.height * scale
        let xCrop = (scaledWidth - viewSize.width) / 2
        let yCrop = (scaledHeight - viewSize.height) / 2

        let imageRect = CGRect(x: normalizedBBox.minX * imageSize.width,
                               y: normalizedBBox.minY * imageSize.height,
                               width: normalizedBBox.width * imageSize.width,
                               height: normalizedBBox.height * imageSize.height)

        let topLeftY = imageSize.height - imageRect.maxY

        return CGRect(x: imageRect.minX * scale - xCrop,
                      y: topLeftY * scale - yCrop,
                      width: imageRect.width * scale,
                      height: imageRect.height * scale)
    }
}

private struct DetectionBox: Shape {
    let observation: VNDetectedObjectObservation

    func path(in rect: CGRect) -> Path {
        Path(CGRect(origin: .zero, size: rect.size))
    }
}
