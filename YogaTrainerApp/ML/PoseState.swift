import SwiftUI
import Combine

class PoseState: ObservableObject {

    @Published var pose: String = "..."
    @Published var holdTime: Double = 0
    @Published var progress: Double = 0
    @Published var completionWord: String?

    private var poseStartTime: Date?
    private var displayPose: String = "..."
    private var window: [String] = []
    private let windowSize = 7
    private let requiredMajority = 4
    private let validThreshold: Double = 0.45
    private var frameCounter = 0

    private let completionWords = [
        "PERFECT", "EXCELLENT", "AMAZING", "BEAUTIFUL", "STRONG",
        "STABLE", "IMPECCABLE", "MASTERFUL", "FLAWLESS", "OUTSTANDING"
    ]

    func shouldClassifyThisFrame(every n: Int) -> Bool {
        frameCounter += 1
        return frameCounter % max(n, 1) == 0
    }

    func updateFromClassifier(label: String, confidence: Double) {
        let isValid = confidence >= validThreshold && label != "unknown" && label != "uncertain" && label != "no_person"

        guard isValid else {
            resetForUncertain()
            return
        }

        window.append(label)
        if window.count > windowSize {
            window.removeFirst()
        }

        guard let majority = majorityPose(), isPoseStable(majority) else { return }

        if majority != displayPose {
            displayPose = majority
            pose = majority
            poseStartTime = Date()
            holdTime = 0
            progress = 0
            completionWord = nil
            return
        }

        if let start = poseStartTime {
            holdTime = Date().timeIntervalSince(start)
            progress = min(holdTime / 10.0, 1.0)

            if progress >= 1.0, completionWord == nil {
                completionWord = completionWords.randomElement()
                DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
                    self.completionWord = nil
                }
            }
        }
    }

    func resetForNoPerson() {
        resetForUncertain()
        pose = "NO_PERSON"
    }

    private func resetForUncertain() {
        window.removeAll()
        displayPose = "..."
        pose = "UNCERTAIN"
        holdTime = 0
        progress = 0
        poseStartTime = nil
        completionWord = nil
    }

    private func majorityPose() -> String? {
        guard !window.isEmpty else { return nil }
        let counts = Dictionary(grouping: window, by: { $0 }).mapValues(
            \.count
        )
        return counts.max(by: { $0.value < $1.value })?.value ?? 0 >= requiredMajority
            ? counts.max(by: { $0.value < $1.value })?.key
            : nil
    }

    private func isPoseStable(_ pose: String) -> Bool {
        let tail = window.suffix(3)
        return tail.filter { $0 == pose }.count >= 2
    }
}
