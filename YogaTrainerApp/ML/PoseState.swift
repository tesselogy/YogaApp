import SwiftUI
import Combine

class PoseState: ObservableObject {

    @Published var pose: String = "..."
    @Published var holdTime: Double = 0
    @Published var progress: Double = 0
    @Published var showCompletion: Bool = false
    @Published var completionWord: String = ""

    var isPoseValid: Bool {
        pose != "..." && pose != "no_person" && pose != "uncertain"
    }

    private var startTime: Date?
    private var timer: Timer?
    private var completionHideWorkItem: DispatchWorkItem?

    private let holdTarget: Double = 10
    private let completionWords = [
        "PERFECT", "EXCELLENT", "AMAZING", "BEAUTIFUL", "STRONG",
        "STABLE", "IMPECCABLE", "MASTERFUL", "FLAWLESS", "OUTSTANDING"
    ]

    func update(newPose: String) {
        if newPose == "uncertain" || newPose == "no_person" || newPose == "..." {
            reset(to: newPose)
            return
        }

        if newPose != pose {
            pose = newPose
            startTime = Date()
            holdTime = 0
            progress = 0
            showCompletion = false
            completionHideWorkItem?.cancel()
            startTimer()
        }
    }

    func reset(to pose: String = "...") {
        self.pose = pose
        startTime = nil
        holdTime = 0
        progress = 0
        showCompletion = false
        completionHideWorkItem?.cancel()
        timer?.invalidate()
        timer = nil
    }

    var easedProgress: Double {
        easeOutCubic(min(max(progress, 0), 1))
    }

    var progressScale: CGFloat {
        progress >= 1 ? 1.2 : 1.0
    }

    var progressColor: Color {
        let p = min(max(progress, 0), 1)
        if p < 0.5 {
            let t = p / 0.5
            return rgbMix(from: (0.73, 0.60, 0.92), to: (0.98, 0.76, 0.63), t: t)
        }
        let t = (p - 0.5) / 0.5
        return rgbMix(from: (0.98, 0.76, 0.63), to: (0.69, 0.95, 0.82), t: t)
    }

    private func startTimer() {
        timer?.invalidate()

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            guard let start = self.startTime else { return }

            self.holdTime = Date().timeIntervalSince(start)
            self.progress = min(self.holdTime / self.holdTarget, 1)

            if self.progress >= 1, !self.showCompletion {
                self.triggerCompletion()
            }
        }
    }

    private func triggerCompletion() {
        completionWord = completionWords.randomElement() ?? "GREAT"
        showCompletion = true

        completionHideWorkItem?.cancel()
        let work = DispatchWorkItem { [weak self] in
            self?.showCompletion = false
        }
        completionHideWorkItem = work
        DispatchQueue.main.asyncAfter(deadline: .now() + 2, execute: work)
    }

    private func easeOutCubic(_ x: Double) -> Double {
        1 - pow(1 - x, 3)
    }

    private func rgbMix(from: (Double, Double, Double),
                        to: (Double, Double, Double),
                        t: Double) -> Color {
        let tt = max(0, min(1, t))
        return Color(
            red: from.0 + (to.0 - from.0) * tt,
            green: from.1 + (to.1 - from.1) * tt,
            blue: from.2 + (to.2 - from.2) * tt
        )
    }
}
