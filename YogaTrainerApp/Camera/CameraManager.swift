import AVFoundation
import Combine

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {

    @Published var currentBuffer: CVPixelBuffer?

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()

    override init() {
        super.init()
        setup()
    }

    private func setup() {
        session.sessionPreset = .high

        guard let device = AVCaptureDevice.default(for: .video),
              let input = try? AVCaptureDeviceInput(device: device)
        else { return }

        session.addInput(input)

        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        session.addOutput(output)

        session.startRunning()
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {

        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        DispatchQueue.main.async {
            self.currentBuffer = buffer
        }
    }
}