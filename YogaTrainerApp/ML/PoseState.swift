import SwiftUI
import Combine

class PoseState: ObservableObject {

    @Published var pose: String = "..."
    @Published var holdTime: Double = 0

    private var startTime: Date?
    private var timer: Timer?

    func update(newPose: String) {
        if newPose != pose {
            pose = newPose
            startTime = Date()
            holdTime = 0
            startTimer()
        }
    }

    private func startTimer() {
        timer?.invalidate()

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
            if let start = self.startTime {
                self.holdTime = Date().timeIntervalSince(start)
            }
        }
    }
}