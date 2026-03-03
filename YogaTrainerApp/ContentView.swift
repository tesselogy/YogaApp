import SwiftUI
import Vision
import Foundation

struct ContentView: View {

    @StateObject var camera = CameraManager()
    @StateObject var poseState = PoseState()

    let focusEngine = FocusTrackingEngine()
    let classifier = ClassificationEngine()

    @State private var trackedObservation: VNDetectedObjectObservation?
    @State private var detectionStatus: String = "Waiting for person..."
    @State private var isProcessingFrame: Bool = false
    @State private var imageSize: CGSize = .zero

    // Pipeline stabilizers
    @State private var lastInferenceAt: Date = .distantPast
    @State private var lockedObservation: VNDetectedObjectObservation?
    @State private var focusLockUntil: Date = .distantPast
    @State private var smoothedROI: CGRect?
    @State private var voteWindow: [String] = []
    @State private var pendingPose: String?
    @State private var pendingPoseSince: Date = .distantPast
    @State private var boxConfidence: Double = 0

    private let processingQueue = DispatchQueue(label: "vision.processing.queue", qos: .userInitiated)

    private let targetInferenceFPS: Double = 10
    private let confidenceThreshold: Double = 0.6
    private let majorityWindowSize: Int = 7
    private let majorityMinVotes: Int = 4
    private let focusLockDuration: TimeInterval = 2.0
    private let poseSwitchHysteresis: TimeInterval = 0.45
    private let roiInterpolationAlpha: CGFloat = 0.35

    var body: some View {

        GeometryReader { geo in
            ZStack {
                CameraPreview(session: camera.captureSession)
                    .scaledToFill()
                    .onChange(of: camera.currentBuffer) { _, newBuffer in
                        guard let buffer = newBuffer else { return }
                        guard !isProcessingFrame else { return }

                        let now = Date()
                        guard now.timeIntervalSince(lastInferenceAt) >= 1.0 / targetInferenceFPS else { return }
                        lastInferenceAt = now

                        imageSize = CGSize(width: CVPixelBufferGetWidth(buffer),
                                           height: CVPixelBufferGetHeight(buffer))
                        isProcessingFrame = true

                        let lockObs = (now < focusLockUntil) ? lockedObservation : nil

                        processingQueue.async {
                            if let lockObs {
                                DispatchQueue.main.async {
                                    classify(buffer: buffer, observation: lockObs)
                                }
                                return
                            }

                            focusEngine.process(buffer: buffer) { obs in
                                DispatchQueue.main.async {
                                    guard let obs else {
                                        trackedObservation = nil
                                        smoothedROI = nil
                                        detectionStatus = "Person not detected"
                                        boxConfidence = 0
                                        voteWindow.removeAll()
                                        pendingPose = nil
                                        poseState.reset(to: "no_person")
                                        isProcessingFrame = false
                                        return
                                    }

                                    lockedObservation = obs
                                    focusLockUntil = Date().addingTimeInterval(focusLockDuration)
                                    classify(buffer: buffer, observation: obs)
                                }
                            }
                        }
                    }

                if let trackedObservation {
                    let rect = toPreviewRect(normalizedBBox: trackedObservation.boundingBox, viewSize: geo.size)

                    DetectionBox(observation: trackedObservation)
                        .stroke(boxColor(for: boxConfidence), lineWidth: boxLineWidth(for: boxConfidence))
                        .frame(width: rect.width, height: rect.height)
                        .position(x: rect.midX, y: rect.midY)
                }

                VStack {
                    Spacer()

                    Text(poseState.pose.uppercased())
                        .font(.system(size: 72, weight: .bold))
                        .foregroundColor(.white)
                        .shadow(color: .black.opacity(0.5), radius: 8, x: 0, y: 3)
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)

                    if poseState.isPoseValid {
                        Text("Hold: \(poseState.holdTime, specifier: "%.1f")s")
                            .font(.title3)
                            .foregroundColor(.white.opacity(0.92))

                        ProgressView(value: poseState.easedProgress)
                            .progressViewStyle(.linear)
                            .tint(poseState.progressColor)
                            .frame(width: min(geo.size.width * 0.5, 420))
                            .scaleEffect(poseState.progressScale)
                            .animation(.interpolatingSpring(stiffness: 210, damping: 13), value: poseState.progressScale)
                    }

                    if poseState.showCompletion {
                        Text(poseState.completionWord)
                            .font(.system(size: 42, weight: .heavy))
                            .foregroundColor(Color(red: 0.69, green: 0.95, blue: 0.82))
                            .scaleEffect(1.08)
                            .opacity(0.95)
                            .transition(.opacity)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: poseState.showCompletion)
                    }

                    Text(detectionStatus)
                        .font(.headline)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 8)
                        .background(Color.black.opacity(0.45))
                        .foregroundColor(.white)
                        .cornerRadius(12)

                    Spacer()

                    HStack {
                        Text("Press Q / Й to quit")
                            .font(.caption)
                            .foregroundColor(.white.opacity(0.9))
                            .padding(10)
                            .background(Color.black.opacity(0.4))
                            .cornerRadius(10)
                        Spacer()
                    }
                    .padding(.leading, 24)
                    .padding(.bottom, 28)
                }
            }
            .ignoresSafeArea()
        }
    }

    private func classify(buffer: CVPixelBuffer, observation: VNDetectedObjectObservation) {
        let rawROI = expandedROI(from: observation.boundingBox)
        let stabilizedROI = stabilizeROI(rawROI)

        trackedObservation = VNDetectedObjectObservation(boundingBox: stabilizedROI)

        classifier.classify(buffer: buffer, regionOfInterest: stabilizedROI) { label, confidence in
            DispatchQueue.main.async {
                boxConfidence = confidence

                guard confidence >= confidenceThreshold, label != "unknown", label != "model_missing" else {
                    detectionStatus = "Uncertain (\(String(format: "%.2f", confidence)))"
                    voteWindow.removeAll()
                    pendingPose = nil
                    poseState.update(newPose: "uncertain")
                    isProcessingFrame = false
                    return
                }

                detectionStatus = "Person detected"
                pushVote(label)
                if let stablePose = resolvedMajorityPose() {
                    applyHysteresis(targetPose: stablePose)
                }

                isProcessingFrame = false
            }
        }
    }

    private func expandedROI(from bbox: CGRect) -> CGRect {
        let widthScale: CGFloat = 2.0
        let heightScale: CGFloat = 2.3
        let newWidth = min(1, bbox.width * widthScale)
        let newHeight = min(1, bbox.height * heightScale)
        let newX = max(0, min(1 - newWidth, bbox.midX - newWidth / 2))
        let newY = max(0, min(1 - newHeight, bbox.midY - newHeight / 2))
        return CGRect(x: newX, y: newY, width: newWidth, height: newHeight)
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

    private func resolvedMajorityPose() -> String? {
        guard voteWindow.count >= majorityMinVotes else { return nil }
        let grouped = Dictionary(grouping: voteWindow, by: { $0 })
        guard let winner = grouped.max(by: { $0.value.count < $1.value.count }) else { return nil }
        return winner.value.count >= majorityMinVotes ? winner.key : nil
    }

    private func applyHysteresis(targetPose: String) {
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

    private func boxColor(for confidence: Double) -> Color {
        let c = max(0, min(1, confidence))
        if c < 0.5 {
            let t = c / 0.5
            return Color(red: 0.73 + (0.98 - 0.73) * t,
                         green: 0.60 + (0.76 - 0.60) * t,
                         blue: 0.92 + (0.63 - 0.92) * t)
        }

        let t = (c - 0.5) / 0.5
        return Color(red: 0.98 + (0.69 - 0.98) * t,
                     green: 0.76 + (0.95 - 0.76) * t,
                     blue: 0.63 + (0.82 - 0.63) * t)
    }

    private func boxLineWidth(for confidence: Double) -> CGFloat {
        CGFloat(2 + max(0, min(1, confidence)) * 4)
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
