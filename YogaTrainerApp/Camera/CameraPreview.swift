import SwiftUI
import CoreImage

struct CameraPreview: View {

    var pixelBuffer: CVPixelBuffer?

    private let context = CIContext(options: [.useSoftwareRenderer: false])

    var body: some View {

        GeometryReader { geo in
            if let buffer = pixelBuffer,
               let cgImage = convertToCGImage(buffer: buffer) {

                Image(decorative: cgImage, scale: 1.0)
                    .resizable()
                    .scaledToFill()
                    .frame(width: geo.size.width, height: geo.size.height)
                    .clipped()
            } else {
                Color.black
            }
        }
    }

    private func convertToCGImage(buffer: CVPixelBuffer) -> CGImage? {
        let ciImage = CIImage(cvPixelBuffer: buffer)
        return context.createCGImage(ciImage, from: ciImage.extent)
    }
}
