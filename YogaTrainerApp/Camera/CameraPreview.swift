import SwiftUI
import AVFoundation

struct CameraPreview: NSViewRepresentable {

    let session: AVCaptureSession

    func makeNSView(context: Context) -> CameraPreviewView {
        let view = CameraPreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateNSView(_ nsView: CameraPreviewView, context: Context) {
        if nsView.previewLayer.session !== session {
            nsView.previewLayer.session = session
        }
    }
}

final class CameraPreviewView: NSView {
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        layer = AVCaptureVideoPreviewLayer()
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        guard let layer = layer as? AVCaptureVideoPreviewLayer else {
            let fallback = AVCaptureVideoPreviewLayer()
            self.layer = fallback
            return fallback
        }
        return layer
    }
}
